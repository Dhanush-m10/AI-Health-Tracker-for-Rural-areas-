import { SymptomReport, AgeGroup, Gender } from "../types";

type ModelProbability = {
  label: string;
  probability: number;
};

type ModelPredictResponse = {
  predicted_label: string;
  probabilities: ModelProbability[];
};

const API_BASE_URL = (import.meta.env.VITE_ML_API_URL || "/api").replace(/\/$/, "");

const FEATURE_NAMES = [
  "fever",
  "cough",
  "headache",
  "nausea",
  "vomiting",
  "fatigue",
  "sore_throat",
  "chills",
  "body_pain",
  "loss_of_appetite",
  "abdominal_pain",
  "diarrhea",
  "sweating",
  "rapid_breathing",
  "dizziness",
] as const;

const FEATURE_KEYWORDS: Record<(typeof FEATURE_NAMES)[number], string[]> = {
  fever: ["fever", "high temperature", "temperature"],
  cough: ["cough", "dry cough", "wet cough"],
  headache: ["headache", "head pain", "migraine"],
  nausea: ["nausea", "nauseous"],
  vomiting: ["vomit", "vomiting", "threw up"],
  fatigue: ["fatigue", "tired", "weakness", "weak"],
  sore_throat: ["sore throat", "throat pain"],
  chills: ["chills", "shivering"],
  body_pain: ["body pain", "body ache", "muscle pain", "joint pain"],
  loss_of_appetite: ["loss of appetite", "no appetite", "not hungry"],
  abdominal_pain: ["abdominal pain", "stomach pain", "belly pain"],
  diarrhea: ["diarrhea", "loose motions", "loose stool"],
  sweating: ["sweating", "sweaty"],
  rapid_breathing: ["rapid breathing", "shortness of breath", "breathless", "breathing fast"],
  dizziness: ["dizziness", "dizzy", "lightheaded"],
};

const CONDITION_DESCRIPTIONS: Record<string, string> = {
  Typhoid: "May involve persistent fever, body weakness, and digestive discomfort.",
  Malaria: "Often linked with fever, chills, sweating, and fatigue in endemic regions.",
  Pneumonia: "Can involve cough, breathing difficulty, fever, and chest discomfort.",
};

function parseDetectedSymptoms(symptomText: string): string[] {
  const normalized = symptomText.toLowerCase();
  const detected: string[] = [];

  for (const feature of FEATURE_NAMES) {
    const keywords = FEATURE_KEYWORDS[feature];
    const found = keywords.some((keyword) => normalized.includes(keyword));
    if (found) {
      detected.push(feature.replace(/_/g, " "));
    }
  }

  return detected;
}

function buildFeaturePayload(symptomText: string): Record<string, number> {
  const normalized = symptomText.toLowerCase();
  const payload: Record<string, number> = {};

  for (const feature of FEATURE_NAMES) {
    const keywords = FEATURE_KEYWORDS[feature];
    payload[feature] = keywords.some((keyword) => normalized.includes(keyword)) ? 1 : 0;
  }

  return payload;
}

function formatConditionDescription(label: string, probability: number): string {
  const base = CONDITION_DESCRIPTIONS[label] || "Prediction based on symptoms recognized by the trained model.";
  return `${base} Model confidence: ${(probability * 100).toFixed(1)}%.`;
}

function buildPrecautions(predictedLabel: string): string[] {
  const common = [
    "Drink clean and safe water regularly.",
    "Take adequate rest and avoid heavy physical work.",
    "Eat light, nutritious meals and avoid outside oily food.",
    "Monitor fever and hydration through the day.",
  ];

  const byCondition: Record<string, string[]> = {
    Typhoid: [
      "Use boiled water only and maintain strict food hygiene.",
      "Avoid raw foods until symptoms improve.",
    ],
    Malaria: [
      "Use mosquito protection, especially at night.",
      "Seek a blood test confirmation at the nearest clinic.",
    ],
    Pneumonia: [
      "Avoid dust and smoke exposure.",
      "Keep breathing warm and stay in a ventilated room.",
    ],
  };

  return [...common, ...(byCondition[predictedLabel] || [])];
}

function buildDoctorWarnings(ageGroup: AgeGroup): string[] {
  const warnings = [
    "High fever lasting more than 2 days.",
    "Breathing difficulty, chest pain, or persistent vomiting.",
    "Severe weakness, confusion, or fainting.",
    "No improvement after basic care.",
  ];

  if (ageGroup === "Child (0-12)" || ageGroup === "Senior (50+)") {
    warnings.push("For children and seniors, consult a doctor early even for mild symptoms.");
  }

  return warnings;
}

export async function analyzeSymptoms(
  symptoms: string,
  ageGroup: AgeGroup,
  gender: Gender,
  location: string
): Promise<SymptomReport> {
  const featurePayload = buildFeaturePayload(symptoms);
  const detectedSymptoms = parseDetectedSymptoms(symptoms);

  const payload = {
    features: featurePayload,
    top_k: 3,
    context: {
      ageGroup,
      gender,
      location,
    },
  };

  const baseCandidates = Array.from(new Set([API_BASE_URL, ""]));
  let lastError = "Model API endpoint is unavailable.";
  let result: ModelPredictResponse | null = null;

  for (const base of baseCandidates) {
    const endpoint = `${base}/predict`;
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (response.ok) {
      result = (await response.json()) as ModelPredictResponse;
      break;
    }

    let detail = "";
    try {
      const errJson = (await response.json()) as { detail?: string };
      detail = errJson?.detail || "";
    } catch {
      detail = (await response.text()).trim();
    }

    if (response.status === 404) {
      lastError = detail || `Endpoint not found at ${endpoint}`;
      continue;
    }

    throw new Error(`Model API error (${response.status}): ${detail || "Unknown error"}`);
  }

  if (!result) {
    throw new Error(`Model API error: ${lastError}`);
  }
  const topCandidates = result.probabilities.length
    ? result.probabilities
    : [{ label: result.predicted_label, probability: 1 }];

  return {
    detectedSymptoms: detectedSymptoms.length ? detectedSymptoms : ["No known symptom keyword detected"],
    possibleConditions: topCandidates.map((item) => ({
      name: item.label,
      description: formatConditionDescription(item.label, item.probability),
    })),
    precautions: buildPrecautions(result.predicted_label),
    whenToSeeDoctor: buildDoctorWarnings(ageGroup),
  };
}

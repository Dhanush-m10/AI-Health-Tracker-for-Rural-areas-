import { GoogleGenAI, Type } from "@google/genai";
import { SymptomReport, AgeGroup, Gender } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

export async function analyzeSymptoms(
  symptoms: string,
  ageGroup: AgeGroup,
  gender: Gender,
  location: string
): Promise<SymptomReport> {
  const model = "gemini-3-flash-preview";
  
  const prompt = `
    You are a helpful health assistant for people in rural areas. 
    Analyze the following symptoms and provide basic health guidance.
    
    User Details:
    - Age Group: ${ageGroup}
    - Gender: ${gender}
    - Location: ${location}
    - Symptoms: ${symptoms}
    
    IMPORTANT: 
    - Provide simple, easy-to-understand advice.
    - Clearly state that this is NOT professional medical advice.
    - Use a supportive and calm tone.
  `;

  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          detectedSymptoms: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: "List of symptoms identified from the user's description"
          },
          possibleConditions: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                name: { type: Type.STRING },
                description: { type: Type.STRING }
              },
              required: ["name", "description"]
            },
            description: "2-3 possible common conditions"
          },
          precautions: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: "Simple home-based precautions and advice"
          },
          whenToSeeDoctor: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: "Warning signs that require immediate medical attention"
          }
        },
        required: ["detectedSymptoms", "possibleConditions", "precautions", "whenToSeeDoctor"]
      }
    }
  });

  const text = response.text;
  if (!text) throw new Error("No response from AI");
  
  return JSON.parse(text) as SymptomReport;
}

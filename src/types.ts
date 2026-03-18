export interface SymptomReport {
  detectedSymptoms: string[];
  possibleConditions: {
    name: string;
    description: string;
  }[];
  precautions: string[];
  whenToSeeDoctor: string[];
}

export interface Hospital {
  name: string;
  distance: string;
  contact: string;
  location: string;
}

export type AgeGroup = 'Child (0-12)' | 'Teen (13-18)' | 'Adult (19-50)' | 'Senior (50+)';
export type Gender = 'Male' | 'Female' | 'Prefer not to say';

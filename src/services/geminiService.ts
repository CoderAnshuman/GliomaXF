import { GoogleGenAI, Type } from "@google/genai";

const getAI = () => {
  const apiKey = import.meta.env?.VITE_GEMINI_API_KEY || process.env.VITE_GEMINI_API_KEY;
  if (!apiKey || apiKey === "MY_GEMINI_API_KEY") {
    console.error("And error in getAI: missing API key", { key: apiKey });
    throw new Error("API Key Missing: Please set VITE_GEMINI_API_KEY in your .env file or AI Studio secrets.");
  }

  console.debug("Gemini API key loaded successfully.");
  return new GoogleGenAI({ apiKey });
};

export async function analyzeMRI(base64Image: string, mimeType: string) {
  try {
    // Convert base64 to File for custom model analysis
    const binaryString = atob(base64Image.split(',')[1]);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    const file = new File([bytes], 'mri-scan.jpg', { type: mimeType });
    
    // Use custom CNN-Transformer model for analysis
    return await analyzeWithCustomModel(file);
  } catch (error) {
    console.error("MRI Analysis Error:", error);
    throw new Error(error instanceof Error ? error.message : "Failed to analyze MRI. Please check your connection.");
  }
}

export async function chatWithGemini(message: string, history: { role: 'user' | 'model'; parts: { text: string }[] }[] = []) {
  try {
    const ai = getAI();
    const chat = ai.chats.create({
      model: "gemini-3-flash-preview", // Switched to Flash for higher rate limits
      config: {
        systemInstruction: "You are Gliomax AI, a medical assistant for neuro-oncologists. You provide information about brain tumor diagnostics (Glioma, Meningioma, Pituitary, and No Tumor), our CNN-Transformer architecture, and clinical performance. Be precise, clinical, and helpful. Use markdown for formatting. If asked about your creators, mention Anshuman Shukla as the Leader from Parul University.",
      },
      history: history,
    });

    const result = await chat.sendMessage({ message });
    return result.text;
  } catch (e: any) {
    console.error("Chat Error:", e);
    if (e.message?.includes("429") || e.message?.includes("RESOURCE_EXHAUSTED")) {
      throw new Error("Rate limit exceeded. Please wait a moment before sending another message.");
    }
    throw new Error(e.message || "Failed to connect to clinical database.");
  }
}

export async function findNearbyResearchCenters(lat: number, lng: number) {
  try {
    const ai = getAI();
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: "Find major neuro-oncology research centers and hospitals specializing in brain tumor treatment near these coordinates.",
      config: {
        tools: [{ googleMaps: {} }],
        toolConfig: {
          retrievalConfig: {
            latLng: {
              latitude: lat,
              longitude: lng
            }
          }
        }
      },
    });
    
    return {
      text: response.text,
      places: response.candidates?.[0]?.groundingMetadata?.groundingChunks || []
    };
  } catch (e: any) {
    console.error("Maps Error:", e);
    throw new Error(e.message || "Failed to search for nearby centers.");
  }
}

const API_URL = "https://anshumanshukla-gliomax-docker.hf.space";

// src/services/modelService.ts
export async function analyzeWithCustomModel(file: File) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    console.log('📤 Sending file to backend:', file.name, file.type, file.size);
    
    const response = await await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    console.log('📥 Response status:', response.status, response.statusText);
    
    const data = await response.json();
    console.log('📥 Response data:', data);

    if (!response.ok) {
      console.error('❌ HTTP error response:', response.status, response.statusText, data);
      throw new Error(data.error || `HTTP error: ${response.status} ${response.statusText}`);
    }

    // Handle validation rejections (status: 200 but validation failed)
    if (data.status === 'rejected') {
      console.warn('⚠️ Validation rejected:', data);
      throw new Error(data.errors?.[0] || 'Validation failed');
    }

    // Success - model returned a prediction
    if (data.status === 'ok' || data.status === 'warn') {
      console.log('✅ Success response received');
      
      // Transform backend response to frontend expected format
      return {
        diagnosis: data.pred_class,
        confidence: data.confidence,
        clinicalSummary: `AI analysis indicates ${data.pred_class} with ${(data.confidence * 100).toFixed(1)}% confidence. ${data.warnings?.length ? 'Warning: ' + data.warnings[0] : ''}`,
        warnings: data.warnings || [],
        allProbabilities: data.probabilities,
        tumorLocation: "Brain", // Default location
        suggestedNextSteps: data.suggestedNextSteps,
        images: data.images,
        inferenceMs: data.inference_ms
      };
    }

    // Handle backend errors
    if (data.status === 'error') {
      console.error('❌ Backend processing error:', data);
      throw new Error(data.message || data.error || 'Backend processing error');
    }

    // If we reach here, the response format is unexpected
    console.error('Unexpected response format:', data);
    throw new Error(data.error || 'Unexpected response format from backend');
  } catch (error) {
    console.error('Error analyzing image:', error);
    throw error;
  }
}

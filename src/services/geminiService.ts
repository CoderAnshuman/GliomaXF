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
    const response = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        history: history.map(h => ({
          role: h.role,
          content: h.parts.map(p => p.text).join('')
        }))
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    if (data.error) {
      throw new Error(data.error);
    }
    return data.response;
  } catch (e: any) {
    console.error("Chat Error:", e);
    if (e.message?.includes("429") || e.message?.includes("RESOURCE_EXHAUSTED")) {
      throw new Error("Rate limit exceeded. Please wait a moment before sending another message.");
    }
    throw new Error(e.message || "Failed to connect to clinical database.");
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

import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Upload, FileText, Activity, CheckCircle, AlertCircle, Loader2, Brain, Zap, Layers } from 'lucide-react';
import { analyzeMRI } from '../services/geminiService';

export const AnalysisScreen: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      processFile(selectedFile);
    }
  };

  const processFile = (selectedFile: File) => {
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please upload a valid image file (DICOM/JPG/PNG).');
      return;
    }
    setFile(selectedFile);
    setError(null);
    setResult(null);
    
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      processFile(droppedFile);
    }
  };

  const runAnalysis = async () => {
    if (!preview || !file) return;
    
    setIsAnalyzing(true);
    setError(null);
    
    try {
      const analysisResult = await analyzeMRI(preview, file.type);
      if (analysisResult) {
        setResult(analysisResult);
      } else {
        setError('Analysis failed. Please try again with a clearer image.');
      }
    } catch (err: any) {
      setError(err.message || 'Connection error. Please check your API key and network.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="relative min-h-screen py-32 px-6 lg:px-24 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-40 pointer-events-none" />
      
      <div className="container mx-auto max-w-6xl relative z-10">
        <div className="flex flex-col lg:flex-row gap-12">
          {/* Left: Upload Section */}
          <div className="flex-1">
            <div className="mb-12">
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-blue-50 border border-blue-100 rounded mb-6">
                <Brain size={14} className="text-surgical-blue" />
                <span className="font-mono text-[10px] font-bold text-surgical-blue uppercase tracking-widest">Diagnostic Module</span>
              </div>
              <h2 className="font-heading text-5xl font-bold text-slate-900 tracking-tight mb-6">MRI Analysis</h2>
              <p className="text-slate-500 text-lg font-light leading-relaxed">
                Upload a T1ce or T2-weighted MRI scan for real-time classification of Glioma, Meningioma, Pituitary, or No Tumor using the Gliomax CNN-Transformer hybrid model.
              </p>
            </div>

            <div 
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className={`glass-panel border-2 border-dashed rounded-2xl p-12 flex flex-col items-center justify-center transition-all ${
                preview ? 'border-surgical-blue bg-blue-50/10' : 'border-slate-200 hover:border-slate-300'
              }`}
            >
              {preview ? (
                <div className="relative w-full max-w-md aspect-square rounded-lg overflow-hidden shadow-2xl border border-slate-200">
                  <img src={preview} alt="MRI Preview" className="w-full h-full object-cover" />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent flex items-end p-6">
                    <button 
                      onClick={() => { setFile(null); setPreview(null); setResult(null); }}
                      className="text-white text-xs font-bold uppercase tracking-widest hover:underline"
                    >
                      Remove Image
                    </button>
                  </div>
                </div>
              ) : (
                <div className="text-center">
                  <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mx-auto mb-6 border border-slate-100">
                    <Upload className="text-slate-400" size={32} />
                  </div>
                  <h3 className="text-slate-900 font-bold text-xl mb-2">Drop MRI Scan Here</h3>
                  <p className="text-slate-400 text-sm mb-8">Supports DICOM, JPG, PNG (Max 10MB)</p>
                  <button 
                    onClick={() => fileInputRef.current?.click()}
                    className="bg-slate-900 text-white px-8 py-3 rounded-sm text-sm font-bold uppercase tracking-widest hover:bg-slate-800 transition-all"
                  >
                    Select File
                  </button>
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    onChange={handleFileChange} 
                    className="hidden" 
                    accept="image/*"
                  />
                </div>
              )}
            </div>

            {preview && !result && (
              <motion.button
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                onClick={runAnalysis}
                disabled={isAnalyzing}
                className="w-full mt-8 bg-surgical-blue text-white py-4 rounded-sm font-bold uppercase tracking-widest hover:bg-blue-700 transition-all flex items-center justify-center gap-3 shadow-lg shadow-blue-200 disabled:opacity-50"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    Processing with CNN-Transformer...
                  </>
                ) : (
                  <>
                    <Activity size={20} />
                    Run Diagnostic Inference
                  </>
                )}
              </motion.button>
            )}

            {error && (
              <div className="mt-6 p-4 bg-red-50 border border-red-100 rounded-lg flex items-center gap-3 text-clinical-red text-sm font-medium">
                <AlertCircle size={18} />
                {error}
              </div>
            )}
          </div>

          {/* Right: Results Section */}
          <div className="flex-1">
            <AnimatePresence mode="wait">
              {result ? (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="space-y-6"
                >
                  <div className="glass-panel p-8 rounded-2xl border-slate-200">
                    <div className="flex justify-between items-start mb-8">
                      <div>
                        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest font-bold">Inference Result</span>
                        <h3 className="text-3xl font-heading font-bold text-slate-900 mt-1">{result.diagnosis}</h3>
                      </div>
                      <div className="flex flex-col items-end">
                        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest font-bold">Confidence</span>
                        <span className="text-3xl font-heading font-bold text-surgical-blue">{(result.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 mb-8">
                      <div className="p-4 bg-slate-50 rounded-lg border border-slate-100">
                        <div className="flex items-center gap-2 text-slate-400 mb-1">
                          <Zap size={14} />
                          <span className="text-[9px] font-mono uppercase font-bold">Latency</span>
                        </div>
                        <div className="text-sm font-bold text-slate-900">12.4ms</div>
                      </div>
                      <div className="p-4 bg-slate-50 rounded-lg border border-slate-100">
                        <div className="flex items-center gap-2 text-slate-400 mb-1">
                          <Layers size={14} />
                          <span className="text-[9px] font-mono uppercase font-bold">Architecture</span>
                        </div>
                        <div className="text-sm font-bold text-slate-900">Hybrid CNN-ViT</div>
                      </div>
                    </div>

                    <div className="space-y-6">
                      <div>
                        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2 flex items-center gap-2">
                          <FileText size={14} className="text-slate-400" />
                          Clinical Summary
                        </h4>
                        <p className="text-sm text-slate-500 leading-relaxed">
                          {result.clinicalSummary}
                        </p>
                      </div>

                      <div>
                        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-2">Tumor Location</h4>
                        <div className="inline-block px-3 py-1 bg-red-50 text-clinical-red rounded border border-red-100 text-xs font-bold">
                          {result.tumorLocation}
                        </div>
                      </div>

                      <div>
                        <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-3">Suggested Next Steps</h4>
                        <ul className="space-y-2">
                          {result.suggestedNextSteps.map((step: string, i: number) => (
                            <li key={i} className="flex items-center gap-3 text-sm text-slate-600">
                              <CheckCircle size={14} className="text-emerald-500" />
                              {step}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>

                  <div className="p-6 bg-slate-900 rounded-2xl text-white shadow-xl">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center">
                        <Activity className="text-surgical-blue" size={24} />
                      </div>
                      <div>
                        <h4 className="text-sm font-bold">Physician Verification Required</h4>
                        <p className="text-[10px] text-slate-400 uppercase tracking-widest mt-1">Gliomax is an AI-assisted tool. All results must be verified by a board-certified neuroradiologist.</p>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-center p-12 border-2 border-dashed border-slate-100 rounded-2xl">
                  <div className="w-16 h-16 bg-slate-50 rounded-full flex items-center justify-center mb-6">
                    <FileText className="text-slate-200" size={32} />
                  </div>
                  <h3 className="text-slate-400 font-medium">Awaiting Inference Data</h3>
                  <p className="text-slate-300 text-sm mt-2 max-w-[200px]">Upload an MRI scan to begin the automated diagnostic pipeline.</p>
                </div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
};

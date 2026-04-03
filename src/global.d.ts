interface AIStudio {
  hasSelectedApiKey(): Promise<boolean>;
  openSelectKey(): Promise<void>;
}

interface Window {
  aistudio?: AIStudio;
}

declare global {
  interface ImportMetaEnv {
    readonly VITE_GEMINI_API_KEY?: string;
  }

  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

export {};

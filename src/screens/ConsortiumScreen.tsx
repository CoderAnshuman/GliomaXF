import React from 'react';
import { motion } from 'motion/react';
import { Brain, Code2, Server, Layout, BrainCircuit, Github } from 'lucide-react';

const researchers = [
  {
    name: "Anshuman Shukla",
    institution: "Parul University",
    role: "ML/AI Engineer",
    icon: <Brain size={24} />,
    image: "src\\assets\\anshumanImg.jpeg",
    github: "https://github.com/CoderAnshuman",
  },
  {
    name: "Foram Thakkar",
    institution: "Parul University",
    role: "UI/UX Developer",
    icon: <Layout size={24} />,
    image: "src\\assets\\foramImg.jpeg",
    github: "https://github.com/foramthakkar",
  },
  {
    name: "Tirth Patel",
    institution: "Parul University",
    role: "Backend Developer",
    icon: <Server size={24} />,
    image: "src\\assets\\tirthImg.jpeg",
    github: "https://github.com/tirthpatel",
  },
  {
    name: "Daksh Patel",
    institution: "Parul University",
    role: "Frontend Engineer",
    icon: <Code2 size={24} />,
    image: "src\\assets\\dakshImg.jpeg",
    github: "https://github.com/dakshpatel",
  },
];

const partners = [
  "https://picsum.photos/seed/jh/200/80?grayscale",
  "https://picsum.photos/seed/mit/200/80?grayscale",
  "https://picsum.photos/seed/charite/200/80?grayscale",
  "https://picsum.photos/seed/stanford/200/80?grayscale",
  "https://picsum.photos/seed/siemens/200/80?grayscale",
];

export const ConsortiumScreen: React.FC = () => {
  return (
    <div className="relative min-h-screen py-32 px-6 lg:px-24 overflow-hidden flex flex-col">
      <div className="absolute inset-0 grid-bg opacity-40 pointer-events-none" />

      <div className="container mx-auto max-w-7xl relative z-10 flex-grow">
        <div className="text-center mb-24">
          <span className="inline-block py-1 px-4 rounded-full bg-slate-100 text-slate-500 text-[10px] font-heading font-bold tracking-[0.2em] uppercase border border-slate-200 mb-6">
            Consortium
          </span>
          <h2 className="text-slate-400 font-heading font-semibold text-sm tracking-[0.3em] uppercase mb-4">
            Developed in Partnership With
          </h2>
          <div className="h-px w-24 bg-slate-300 mx-auto mt-8" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-32">
          {researchers.map((person, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="group h-[320px] cursor-pointer"
              style={{ perspective: '1000px' }}
            >
              <div
                className="relative w-full h-full transition-transform duration-700"
                style={{ transformStyle: 'preserve-3d' }}
                onMouseEnter={e => (e.currentTarget.style.transform = 'rotateY(180deg)')}
                onMouseLeave={e => (e.currentTarget.style.transform = 'rotateY(0deg)')}
              >
                {/* FRONT — Role Details */}
                <div
                  className="absolute inset-0 rounded-xl glass-panel border border-slate-200 p-8 flex flex-col justify-between"
                  style={{ backfaceVisibility: 'hidden' }}
                >
                  <div>
                    <div className="w-14 h-14 mb-6 rounded-lg bg-slate-50 flex items-center justify-center border border-slate-200 text-surgical-blue">
                      {person.icon}
                    </div>
                    <h3 className="text-slate-900 font-heading font-bold text-xl mb-1">{person.name}</h3>
                    <p className="text-slate-500 text-sm font-medium">{person.institution}</p>
                  </div>
                  <div className="pt-6 border-t border-slate-100">
                    <span className="text-slate-400 font-mono text-[9px] uppercase tracking-widest block mb-2">Role</span>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-700 font-mono text-[10px] bg-slate-100 px-3 py-1.5 rounded border border-slate-200 font-bold">
                        {person.role}
                      </span>
                      {/* Hint icon — non-interactive on front */}
                      <Github size={16} className="text-slate-300" />
                    </div>
                  </div>
                </div>

                {/* BACK — GitHub Profile Photo */}
                <div
                  className="absolute inset-0 rounded-xl overflow-hidden border border-slate-200"
                  style={{ backfaceVisibility: 'hidden', transform: 'rotateY(180deg)' }}
                >
                  <img
                    src={person.image}
                    alt={person.name}
                    className="w-full h-full object-cover"
                    referrerPolicy="no-referrer"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent" />
                  <div className="absolute bottom-4 left-4 right-4 flex items-end justify-between">
                    <div>
                      <h3 className="text-white font-heading font-bold text-xl">{person.name}</h3>
                      <p className="text-white/70 text-sm">{person.institution}</p>
                    </div>
                    <a
                      href={person.github}
                      target="_blank"
                      rel="noreferrer"
                      className="text-white/80 hover:text-white transition-colors mb-1"
                      onClick={e => e.stopPropagation()}
                    >
                      <Github size={20} />
                    </a>
                  </div>
                </div>

              </div>
            </motion.div>
          ))}
        </div>

        <div className="border-t border-slate-200 pt-20 pb-12">
          <p className="text-center text-slate-400 text-[10px] font-mono mb-12 uppercase tracking-[0.4em] font-bold">
            Trust & Verification Partners
          </p>
          <div className="flex flex-wrap justify-center items-center gap-8 md:gap-16 lg:gap-24 opacity-40 grayscale hover:grayscale-0 transition-all duration-700">
            {partners.map((url, i) => (
              <img key={i} src={url} alt="Partner Logo" className="h-8 md:h-10 object-contain" referrerPolicy="no-referrer" />
            ))}
          </div>
        </div>
      </div>

      <footer className="mt-auto pt-12 border-t border-slate-200/60">
        <div className="container mx-auto max-w-7xl flex flex-col md:flex-row justify-between items-center gap-8 text-slate-400 text-[10px] font-mono uppercase tracking-widest font-bold">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2 text-slate-900">
              <BrainCircuit size={16} />
              <span>GLIOMAX</span>
            </div>
            <span className="hidden md:block w-px h-4 bg-slate-200" />
            <p>© 2026 GliomaX Precision Glass. All rights reserved.</p>
          </div>
          <div className="flex items-center gap-12">
            <a href="https://github.com/CoderAnshuman/GliomaXF.git" target="_blank" rel="noreferrer" className="hover:text-slate-900 transition-colors">Documentation</a>
            <a href="https://github.com/CoderAnshuman/GliomaXF.git" target="_blank" rel="noreferrer" className="hover:text-slate-900 transition-colors">GitHub</a>
            <a href="mailto:anshumanmshukla@gmail.com">
  Contact Support
</a>
          </div>
        </div>
      </footer>
    </div>
  );
};
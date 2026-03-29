import { motion } from 'framer-motion';
import { Layers, Database, ShieldCheck, Cpu } from 'lucide-react';

const About = () => {
    return (
        <div className="container" style={{ padding: '4rem 2rem' }}>
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                style={{ maxWidth: '800px', margin: '0 auto' }}
            >
                <h1 style={{ fontSize: '3.5rem', marginBottom: '1rem', textAlign: 'center' }}>
                    Inside the <span className="text-gradient">Engine</span>
                </h1>
                <p style={{ fontSize: '1.2rem', color: 'var(--color-text-secondary)', textAlign: 'center', marginBottom: '4rem' }}>
                    MiniGPT is an educational masterpiece designed to make Large Language Models transparent.
                </p>

                <div className="glass-panel" style={{ padding: '3rem', marginBottom: '3rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem' }}>
                        <Layers color="var(--color-brand-primary)" size={32} />
                        <h2 style={{ fontSize: '2rem' }}>Architectural Specs</h2>
                    </div>
                    <p style={{ color: 'var(--color-text-secondary)', marginBottom: '2rem', fontSize: '1.1rem' }}>
                        MiniGPT is heavily inspired by Llama 3 and Mistral architectures. We stripped away the abstraction of deep learning frameworks to expose the raw mathematics.
                    </p>

                    <ul style={{ listStyle: 'none', display: 'grid', gap: '1rem' }}>
                        {[
                            { title: "Rotary Positional Embeddings (RoPE)", desc: "Replaces absolute positional embeddings for better length generalization." },
                            { title: "Grouped Query Attention (GQA)", desc: "Reduces memory bandwidth compared to Multi-Head Attention." },
                            { title: "SwiGLU Activation", desc: "A variant of GLU with Swish (SiLU) used in LLaMA." },
                            { title: "RMSNorm", desc: "Computationally cheaper alternative to LayerNorm." }
                        ].map((item, i) => (
                            <li key={i} style={{ padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', borderLeft: '4px solid var(--color-brand-secondary)' }}>
                                <strong style={{ display: 'block', color: 'white', marginBottom: '0.25rem' }}>{item.title}</strong>
                                <span style={{ color: 'var(--color-text-secondary)' }}>{item.desc}</span>
                            </li>
                        ))}
                    </ul>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                    <div className="glass-panel" style={{ padding: '2rem' }}>
                        <Database color="var(--color-brand-tertiary)" size={28} style={{ marginBottom: '1rem' }} />
                        <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Custom BPE Tokenizer</h3>
                        <p style={{ color: 'var(--color-text-secondary)' }}>
                            Built identically to GPT-4's tokenizer, featuring regex splitting and byte-level fallback to handle out-of-vocabulary characters perfectly.
                        </p>
                    </div>

                    <div className="glass-panel" style={{ padding: '2rem' }}>
                        <Cpu color="var(--color-brand-secondary)" size={28} style={{ marginBottom: '1rem' }} />
                        <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>KV Caching</h3>
                        <p style={{ color: 'var(--color-text-secondary)' }}>
                            A highly optimized ring-buffer KV Cache avoids recomputing past tokens, ensuring generation speeds of ~42ms/token purely in Python.
                        </p>
                    </div>
                </div>
            </motion.div>
        </div>
    );
};

export default About;

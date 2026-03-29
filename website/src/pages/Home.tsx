import { Link } from 'react-router-dom';
import { color, motion } from 'framer-motion';
import { Terminal, Zap, Cpu, Code2 } from 'lucide-react';

const Home = () => {
    return (
        <div className="container" style={{ padding: '4rem 2rem', textAlign: 'center' }}>
            <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
            >
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '6px 16px', background: 'rgba(109, 40, 217, 0.2)', borderRadius: '100px', marginBottom: '2rem', border: '1px solid rgba(109, 40, 217, 0.4)' }}>
                    <Zap size={16} color="var(--color-brand-secondary)" />
                    <span style={{ fontSize: '0.9rem', fontWeight: 600, color: '#e2d4ff' }}>MiniGPT Engine v1.0 is Live</span>
                </div>

                <h1 style={{ fontSize: 'clamP(3rem, 8vw, 5rem)', marginBottom: '1.5rem', letterSpacing: '-0.03em' }}>
                    The <span className="text-gradient">NumPy-Only</span><br />
                    LLM Inference Engine.
                </h1>

                <p style={{ fontSize: '1.25rem', color: 'var(--color-text-secondary)', maxWidth: '700px', margin: '0 auto 3rem', lineHeight: 1.8 }}>
                    Demystifying the "black box" of Transformers. MiniGPT implements the Llama 3 architecture
                    completely from scratch using nothing but Python and NumPy arrays.
                </p>

                <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginBottom: '5rem' }}>
                    <Link to="/chat" className="btn" style={{ color: 'white' }}>
                        <Terminal size={20} />
                        Try the Engine
                    </Link>
                    <Link to="/about" className="btn-outline" style={{ color: 'white' }}>
                        Technical Details
                    </Link>
                </div>
            </motion.div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', textAlign: 'left' }}>
                <motion.div
                    className="glass-panel"
                    style={{ padding: '2rem' }}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                >
                    <div style={{ width: '50px', height: '50px', borderRadius: '12px', background: 'rgba(236, 72, 153, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.5rem' }}>
                        <Code2 color="var(--color-brand-secondary)" size={24} />
                    </div>
                    <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Zero Dependencies</h3>
                    <p style={{ color: 'var(--color-text-secondary)' }}>No PyTorch. No TensorFlow. Just pure, unadulterated matrix multiplication via standard NumPy.</p>
                </motion.div>

                <motion.div
                    className="glass-panel"
                    style={{ padding: '2rem' }}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.4 }}
                >
                    <div style={{ width: '50px', height: '50px', borderRadius: '12px', background: 'rgba(59, 130, 246, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.5rem' }}>
                        <Cpu color="var(--color-brand-tertiary)" size={24} />
                    </div>
                    <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Modern Architecture</h3>
                    <p style={{ color: 'var(--color-text-secondary)' }}>Features RoPE embeddings, Grouped Query Attention, SwiGLU activations, and KV Caching.</p>
                </motion.div>

                <motion.div
                    className="glass-panel"
                    style={{ padding: '2rem' }}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.6 }}
                >
                    <div style={{ width: '50px', height: '50px', borderRadius: '12px', background: 'rgba(109, 40, 217, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.5rem' }}>
                        <Zap color="var(--color-brand-primary)" size={24} />
                    </div>
                    <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>High Performance</h3>
                    <p style={{ color: 'var(--color-text-secondary)' }}>Optimized with static memory allocation and ring-buffer caching for sub-50ms inference times.</p>
                </motion.div>
            </div>
        </div>
    );
};

export default Home;

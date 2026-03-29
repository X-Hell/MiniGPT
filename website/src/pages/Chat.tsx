import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Send, Bot, User, Loader2 } from 'lucide-react';

interface Message {
    id: number;
    role: 'user' | 'assistant';
    content: string;
}

const Chat = () => {
    const [messages, setMessages] = useState<Message[]>([
        { id: 1, role: 'assistant', content: 'Hello! I am MiniGPT. I run entirely on NumPy! How can I help you today?' }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage = input.trim();
        setInput('');
        setMessages(prev => [...prev, { id: Date.now(), role: 'user', content: userMessage }]);
        setIsLoading(true);

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMessage, max_tokens: 150 })
            });

            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();

            setMessages(prev => [...prev, {
                id: Date.now(),
                role: 'assistant',
                content: data.reply
            }]);
        } catch (error) {
            console.error(error);
            setMessages(prev => [...prev, {
                id: Date.now(),
                role: 'assistant',
                content: "Sorry, I couldn't reach the inference backend. Is the FastAPI server running on port 8000?"
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="container" style={{ padding: '2rem', height: 'calc(100vh - 80px)', display: 'flex', flexDirection: 'column' }}>
            <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '1rem' }}>

                {/* Chat Messages Area */}
                <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    {messages.map((msg) => (
                        <motion.div
                            key={msg.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            style={{
                                display: 'flex',
                                gap: '1rem',
                                alignItems: 'flex-start',
                                flexDirection: msg.role === 'user' ? 'row-reverse' : 'row'
                            }}
                        >
                            <div style={{
                                width: '40px', height: '40px', borderRadius: '50%',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                background: msg.role === 'user' ? 'var(--gradient-primary)' : 'rgba(255,255,255,0.05)',
                                border: msg.role === 'assistant' ? '1px solid var(--color-border)' : 'none',
                                flexShrink: 0
                            }}>
                                {msg.role === 'user' ? <User size={20} color="white" /> : <Bot size={20} color="var(--color-brand-secondary)" />}
                            </div>

                            <div style={{
                                background: msg.role === 'user' ? 'rgba(109, 40, 217, 0.15)' : 'rgba(255,255,255,0.03)',
                                padding: '1rem 1.5rem',
                                borderRadius: '1rem',
                                borderTopRightRadius: msg.role === 'user' ? '0' : '1rem',
                                borderTopLeftRadius: msg.role === 'assistant' ? '0' : '1rem',
                                border: '1px solid var(--color-border)',
                                maxWidth: '80%',
                                lineHeight: 1.6
                            }}>
                                {msg.content}
                            </div>
                        </motion.div>
                    ))}

                    {isLoading && (
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                            <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--color-border)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                <Bot size={20} color="var(--color-brand-secondary)" />
                            </div>
                            <div style={{ padding: '1rem 1.5rem', borderRadius: '1rem', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--color-border)' }}>
                                <Loader2 className="lucide-spin" size={20} color="var(--color-brand-secondary)" style={{ animation: 'spin 2s linear infinite' }} />
                            </div>
                        </motion.div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div style={{ padding: '1rem', borderTop: '1px solid var(--color-border)', marginTop: 'auto' }}>
                    <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '1rem' }}>
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask MiniGPT something..."
                            disabled={isLoading}
                            style={{
                                flex: 1,
                                padding: '1rem 1.5rem',
                                borderRadius: '100px',
                                border: '1px solid var(--color-border)',
                                background: 'rgba(0,0,0,0.3)',
                                color: 'white',
                                fontSize: '1rem',
                                outline: 'none',
                            }}
                        />
                        <button
                            type="submit"
                            disabled={isLoading || !input.trim()}
                            className="btn"
                            style={{ width: '56px', height: '56px', padding: 0, borderRadius: '50%' }}
                        >
                            <Send size={20} style={{ transform: 'translateX(-2px) translateY(2px)' }} />
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default Chat;

/* Add to global css implicitly for spin animation if needed */

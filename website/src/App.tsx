import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import About from './pages/About';
import Chat from './pages/Chat';

function App() {
  return (
    <>
      {/* Ambient animated gradient backgrounds */}
      <div className="ambient-bg">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
        <div className="blob blob-3"></div>
      </div>

      <Navbar />

      <main style={{ paddingTop: '80px' }}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/chat" element={<Chat />} />
        </Routes>
      </main>

      <footer className="footer">
        <p>© 2026 Elsoro Technologies. The NumPy-Only LLM Inference Engine.</p>
      </footer>
    </>
  );
}

export default App;

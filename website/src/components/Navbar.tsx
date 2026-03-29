import { Link, useLocation } from 'react-router-dom';
import { Cpu } from 'lucide-react';

const Navbar = () => {
    const location = useLocation();

    const getLinkClass = (path: string) => {
        return `nav-link ${location.pathname === path ? 'active' : ''}`;
    }

    return (
        <nav className="navbar">
            <Link to="/" className="nav-brand">
                <Cpu size={28} color="var(--color-brand-secondary)" />
                MiniGPT
            </Link>
            <div className="nav-links">
                <Link to="/" className={getLinkClass('/')}>Home</Link>
                <Link to="/about" className={getLinkClass('/about')}>About Engine</Link>
                <Link to="/chat" className="btn" style={{ padding: '0.5rem 1.25rem', color: 'white' }}>Try Chat</Link>
            </div>
        </nav>
    );
};

export default Navbar;

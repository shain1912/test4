import { useState } from 'react';
import { LayoutDashboard, MessageSquare } from 'lucide-react';
import ChatView from './components/ChatView';
import DashboardView from './components/DashboardView';

function App() {
  const [activeTab, setActiveTab] = useState('chat');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 flex flex-col font-sans">
      {/* Dynamic Glassmorphism Navbar */}
      <nav className="sticky top-0 z-50 glass-panel border-b border-slate-800/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-400 flex items-center justify-center flex-shrink-0 animate-pulse-slow shadow-lg shadow-blue-500/20">
                <span className="text-lg">🏙️</span>
              </div>
              <h1 className="text-xl font-semibold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 hidden sm:block">
                Busan Walkability
              </h1>
            </div>

            <div className="flex space-x-1 p-1 bg-slate-900/50 rounded-lg border border-slate-800">
              <button
                onClick={() => setActiveTab('chat')}
                className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all duration-300 ${activeTab === 'chat'
                    ? 'bg-blue-600/20 text-blue-400 shadow-[0_0_15px_rgba(59,130,246,0.2)]'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
              >
                <MessageSquare size={18} />
                <span className="font-medium text-sm">인터뷰</span>
              </button>
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all duration-300 ${activeTab === 'dashboard'
                    ? 'bg-purple-600/20 text-purple-400 shadow-[0_0_15px_rgba(168,85,247,0.2)]'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
              >
                <LayoutDashboard size={18} />
                <span className="font-medium text-sm">대시보드</span>
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 w-full max-w-7xl mx-auto p-4 sm:p-6 lg:p-8 flex flex-col h-[calc(100vh-4rem)]">
        {activeTab === 'chat' ? <ChatView /> : <DashboardView />}
      </main>
    </div>
  );
}

export default App;

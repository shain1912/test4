import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Info } from 'lucide-react';
import axios from 'axios';

export default function ChatView() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [sessionId, setSessionId] = useState(null);
    const [isComplete, setIsComplete] = useState(false);
    const [suggestedReplies, setSuggestedReplies] = useState([]);
    const [issuesCount, setIssuesCount] = useState(0);

    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, loading]);

    // Initial greeting
    useEffect(() => {
        const startChat = async () => {
            try {
                setLoading(true);
                const res = await axios.get(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/chat/start`);
                setMessages([{ role: 'assistant', content: res.data.response }]);
                setSessionId(res.data.session_id);
            } catch (err) {
                setMessages([{ role: 'assistant', content: "서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인해주세요." }]);
            } finally {
                setLoading(false);
            }
        };
        startChat();
    }, []);

    const handleSend = async (text) => {
        if (!text.trim() || isComplete) return;

        const userMessage = text;
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
        setLoading(true);
        setSuggestedReplies([]);

        try {
            const res = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/chat/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: userMessage,
                    session_id: sessionId
                })
            });

            if (!res.ok) throw new Error("Network response was not ok");

            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let aiMessage = '';

            // 껍데기 메시지 먼저 추가
            setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6);
                        try {
                            const data = JSON.parse(dataStr);
                            if (data.type === 'chunk') {
                                aiMessage += data.text;
                                setMessages(prev => {
                                    const newMsgs = [...prev];
                                    newMsgs[newMsgs.length - 1].content = aiMessage;
                                    return newMsgs;
                                });
                            } else if (data.type === 'complete') {
                                setIsComplete(data.is_complete);
                                setIssuesCount(data.collected_issues_count || 0);
                            } else if (data.type === 'error') {
                                console.error('Streaming error:', data.error);
                            }
                        } catch (e) {
                            // JSON 파싱 에러는 무시 (잘린 청크일 가능성)
                        }
                    }
                }
            }

        } catch (err) {
            console.error(err);
            setMessages(prev => {
                const newMsgs = [...prev];
                // 만약 빈 껍데기가 이미 추가되었다면 내용을 바꾸고, 아니면 새로 추가
                if (newMsgs[newMsgs.length - 1].role === 'assistant' && newMsgs[newMsgs.length - 1].content === '') {
                    newMsgs[newMsgs.length - 1].content = "응답을 받는 중 오류가 발생했습니다.";
                } else {
                    newMsgs.push({ role: 'assistant', content: "응답을 받는 중 오류가 발생했습니다." });
                }
                return newMsgs;
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex-1 flex gap-6 h-full w-full">
            {/* Sidebar for Interview Status (Optional in UI, but good for UX) */}
            <div className="hidden lg:flex w-80 flex-col gap-4">
                <div className="glass-panel rounded-2xl p-6 h-full border border-slate-700/50 flex flex-col items-center justify-center text-center">
                    <div className="w-16 h-16 rounded-full bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mb-4">
                        <Info className="text-blue-400" size={32} />
                    </div>
                    <h3 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400 mb-2">
                        AI 인터뷰 진행현황
                    </h3>
                    <p className="text-slate-400 text-sm mb-6">
                        시민의 목소리를 데이터로 변환합니다.
                    </p>

                    <div className="w-full bg-slate-800/50 rounded-xl p-4 border border-slate-700/50 shadow-inner">
                        <div className="text-3xl font-black text-white mb-1"><span className="text-indigo-400">{issuesCount}</span>건</div>
                        <div className="text-xs text-slate-400 font-medium uppercase tracking-wider">현재까지 수집된 이슈</div>
                    </div>

                    {isComplete && (
                        <div className="mt-8 p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-xl w-full text-emerald-400 text-sm font-medium">
                            인터뷰가 성공적으로 종료되었습니다. 대시보드 탭에서 분석 결과를 확인하세요.
                        </div>
                    )}
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col glass-panel rounded-2xl border border-slate-700/50 overflow-hidden shadow-2xl relative">
                {/* Chat Messages */}
                <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
                    {messages.map((msg, idx) => (
                        <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} group`}>
                            <div
                                className={`max-w-[85%] md:max-w-[75%] rounded-2xl px-5 py-4 ${msg.role === 'user'
                                    ? 'bg-blue-600 text-white rounded-br-sm shadow-[0_5px_15px_-3px_rgba(37,99,235,0.3)] border border-blue-500/50'
                                    : 'bg-slate-800/80 text-slate-100 rounded-bl-sm shadow-[0_5px_15px_-3px_rgba(0,0,0,0.3)] border border-slate-700 backdrop-blur-md'
                                    } transition-transform duration-300 transform outline-none`}
                            >
                                <div className="flex items-center gap-2 mb-1">
                                    <span className="text-[10px] font-bold tracking-wider text-opacity-70 uppercase">
                                        {msg.role === 'user' ? '시민' : 'AI 인터뷰어'}
                                    </span>
                                </div>
                                <div className="leading-relaxed whitespace-pre-wrap font-light">{msg.content}</div>
                            </div>
                        </div>
                    ))}

                    {loading && (!messages.length || messages[messages.length - 1].role === 'user' || messages[messages.length - 1].content === '') && (
                        <div className="flex justify-start">
                            <div className="bg-slate-800/80 rounded-2xl rounded-bl-sm px-5 py-4 border border-slate-700 shadow-sm flex items-center gap-3">
                                <Loader2 className="animate-spin text-blue-400" size={18} />
                                <span className="text-slate-400 text-sm animate-pulse">분석 및 응답 생성 중...</span>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Suggested Replies */}
                {!isComplete && suggestedReplies.length > 0 && (
                    <div className="px-6 py-3 flex gap-2 overflow-x-auto bg-slate-900/50 border-t border-slate-800/50">
                        {suggestedReplies.map((reply, idx) => (
                            <button
                                key={idx}
                                onClick={() => handleSend(reply)}
                                className="whitespace-nowrap px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm rounded-full border border-slate-600 transition-colors shadow-sm"
                            >
                                {reply}
                            </button>
                        ))}
                    </div>
                )}

                {/* Input Area */}
                <div className="p-4 bg-slate-900 border-t border-slate-800/80">
                    <form
                        onSubmit={(e) => { e.preventDefault(); handleSend(input); }}
                        className="flex gap-3 max-w-4xl mx-auto items-end relative"
                    >
                        <textarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder={isComplete ? "인터뷰가 종료되었습니다." : "이곳에 의견을 입력해주세요..."}
                            disabled={isComplete || loading}
                            className="flex-1 bg-slate-800 border-0 rounded-2xl px-5 py-4 text-slate-100 placeholder:text-slate-500 focus:ring-2 focus:ring-blue-500 focus:outline-none resize-none min-h-[60px] max-h-32 transition-shadow shadow-inner"
                            rows={1}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSend(input);
                                }
                            }}
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || isComplete || loading}
                            className="bg-blue-600 hover:bg-blue-500 text-white rounded-2xl w-14 h-14 flex items-center justify-center transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_15px_-3px_rgba(37,99,235,0.4)] hover:shadow-[0_0_20px_-3px_rgba(37,99,235,0.6)] flex-shrink-0"
                        >
                            {loading ? <Loader2 className="animate-spin" size={24} /> : <Send size={24} className="ml-1" />}
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}

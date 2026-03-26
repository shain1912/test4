import { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { LayoutDashboard, Users, AlertTriangle, Layers, Loader2 } from 'lucide-react';

export default function DashboardView() {
    const [stats, setStats] = useState(null);
    const [tsneData, setTsneData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [statsRes, tsneRes] = await Promise.all([
                    axios.get(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/dashboard/stats`),
                    axios.get(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/dashboard/tsne`)
                ]);
                setStats(statsRes.data);
                setTsneData(tsneRes.data);
            } catch (err) {
                console.error("Dashboard data load error:", err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    if (loading) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center p-8 text-slate-400">
                <Loader2 className="w-12 h-12 text-blue-500 animate-spin mb-4" />
                <h2 className="text-xl font-medium text-slate-200">데이터를 분석 중입니다...</h2>
                <p className="mt-2 text-sm text-center">AI가 수집된 모든 인터뷰를 읽고, 의미망(Semantic Network)를 구성하여 3차원 공간에 시각화합니다.</p>
            </div>
        );
    }

    if (!stats?.total_interviews) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center p-8 text-slate-400 border border-dashed border-slate-700/50 rounded-2xl glass-panel">
                <LayoutDashboard className="w-16 h-16 text-slate-500 mb-4 opacity-50" />
                <h2 className="text-xl font-medium text-slate-200">아직 수집된 데이터가 없습니다</h2>
                <p className="mt-2 text-sm">인터뷰 탭에서 대화를 진행하여 의견을 수집해주세요.</p>
            </div>
        );
    }

    // Formatting for Recharts
    const catData = Object.entries(stats.category_distribution || {}).map(([name, value]) => ({ name, value }));
    const locData = Object.entries(stats.location_distribution || {}).map(([name, value]) => ({ name, value })).sort((a, b) => b.value - a.value).slice(0, 5);
    const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b'];

    return (
        <div className="flex-1 overflow-y-auto space-y-6 pb-20 w-full font-sans">

            {/* Metric Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                    { title: "총 수집 의견", value: `${stats.total_interviews}건`, icon: <Users size={20} />, color: "text-blue-400", bg: "bg-blue-500/10" },
                    { title: "참여 세션", value: `${stats.total_sessions}명`, icon: <LayoutDashboard size={20} />, color: "text-purple-400", bg: "bg-purple-500/10" },
                    { title: "평균 심각도", value: `${stats.avg_severity}/4.0`, icon: <AlertTriangle size={20} />, color: "text-rose-400", bg: "bg-rose-500/10" },
                    { title: "세션당 평균", value: `${stats.issues_per_session}건`, icon: <Layers size={20} />, color: "text-amber-400", bg: "bg-amber-500/10" }
                ].map((item, i) => (
                    <div key={i} className="glass-panel p-5 rounded-2xl border border-slate-700/50 flex items-center gap-4 transition-transform hover:translate-y-[-2px]">
                        <div className={`p-3 rounded-xl ${item.bg}`}>
                            <div className={item.color}>{item.icon}</div>
                        </div>
                        <div>
                            <p className="text-sm text-slate-400 font-medium mb-1">{item.title}</p>
                            <h4 className="text-2xl font-bold text-white tracking-tight">{item.value}</h4>
                        </div>
                    </div>
                ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Category Chart */}
                <div className="glass-panel p-6 rounded-2xl border border-slate-700/50 flex flex-col h-80">
                    <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-6 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                        카테고리별 분포
                    </h3>
                    <div className="flex-1 w-full relative">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={catData} layout="vertical" margin={{ top: 0, right: 0, left: 30, bottom: 0 }}>
                                <XAxis type="number" hide />
                                <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} width={100} />
                                <Tooltip
                                    cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)' }}
                                />
                                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                    {catData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Location Chart */}
                <div className="glass-panel p-6 rounded-2xl border border-slate-700/50 flex flex-col h-80">
                    <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-6 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-purple-500"></span>
                        주요 불편 위치 (Top 5)
                    </h3>
                    <div className="flex-1 w-full relative">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={locData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                <YAxis hide />
                                <Tooltip
                                    cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                                />
                                <Bar dataKey="value" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* 3D Semantic Network Map */}
            {tsneData && !tsneData.error && tsneData.scatter_data && (
                <div className="glass-panel p-6 rounded-2xl border border-slate-700/50 overflow-hidden w-full relative group">
                    <div className="absolute top-6 left-6 z-10 pointer-events-none">
                        <h3 className="text-lg font-bold text-white mb-1 flex items-center gap-2">
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-emerald-400">AI 의미 공간 (Semantic Space)</span>
                        </h3>
                        <p className="text-xs text-slate-400 max-w-xs leading-relaxed">자연어 임베딩 행렬을 3차원 축소하여 유사한 의미를 가진 의견들을 군집화했습니다. 점에 마우스를 올려 자세한 의견을 확인하세요.</p>
                    </div>

                    <div className="w-full h-[500px] mt-4 rounded-xl overflow-hidden bg-slate-950/50 border border-slate-800 shadow-inner">
                        <Plot
                            data={[
                                {
                                    type: 'scatter3d',
                                    mode: 'markers',
                                    x: tsneData.scatter_data.x,
                                    y: tsneData.scatter_data.y,
                                    z: tsneData.scatter_data.z,
                                    text: tsneData.scatter_data.hover_texts,
                                    hoverinfo: 'text',
                                    marker: {
                                        size: 6,
                                        color: tsneData.scatter_data.severities,
                                        colorscale: 'Viridis',
                                        opacity: 0.8,
                                        line: { width: 0 }
                                    }
                                }
                            ]}
                            layout={{
                                autosize: true,
                                margin: { l: 0, r: 0, b: 0, t: 0 },
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                plot_bgcolor: 'rgba(0,0,0,0)',
                                scene: {
                                    xaxis: { visible: false, showgrid: false },
                                    yaxis: { visible: false, showgrid: false },
                                    zaxis: { visible: false, showgrid: false },
                                    camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
                                }
                            }}
                            useResizeHandler={true}
                            style={{ width: '100%', height: '100%' }}
                            config={{ displayModeBar: false }}
                        />
                    </div>
                </div>
            )}

            {/* 3D Topics Detailed Cards */}
            {tsneData && tsneData.topics && (
                <div className="mt-8">
                    <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 px-2">주요 감지 패턴</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {tsneData.topics.map((topic, i) => (
                            <div key={i} className="glass-panel p-5 rounded-2xl border border-slate-700/50 flex flex-col hover:border-blue-500/30 transition-colors">
                                <div className="flex justify-between items-start mb-4">
                                    <h4 className="text-base font-bold text-slate-100">{topic.label}</h4>
                                    <span className="px-2.5 py-1 bg-slate-800 text-slate-300 text-xs rounded-full font-medium border border-slate-700">
                                        {topic.count}건
                                    </span>
                                </div>
                                <div className="flex items-center gap-2 mb-4 bg-slate-900/40 p-2 rounded-lg">
                                    <span className="text-xs text-slate-400 uppercase font-medium">평균 심각도</span>
                                    <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full ${topic.avg_severity >= 3 ? 'bg-rose-500' : topic.avg_severity >= 2 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                                            style={{ width: `${(topic.avg_severity / 4) * 100}%` }}
                                        ></div>
                                    </div>
                                    <span className="text-sm font-black text-white">{topic.avg_severity.toFixed(1)}</span>
                                </div>
                                <div className="space-y-2 mt-auto">
                                    {topic.sample_issues.map((issue, j) => (
                                        <div key={j} className="text-sm text-slate-400 italic font-light line-clamp-2 border-l-2 border-slate-700 pl-3">
                                            "{issue}"
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

        </div>
    );
}

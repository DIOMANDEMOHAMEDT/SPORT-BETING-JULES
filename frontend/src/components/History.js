import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const API_URL = 'http://localhost:8000';

const History = () => {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                setLoading(true);
                // Note: In a real app, the JWT token would be included here as well.
                const response = await axios.get(`${API_URL}/get_history`);
                const formattedHistory = response.data.history.map(item => ({
                    ...item,
                    date: new Date(item.date).toLocaleDateString(),
                    profit: parseFloat(item.profit).toFixed(2),
                    bankroll_after_bet: parseFloat(item.bankroll_after_bet).toFixed(2)
                }));
                setHistory(formattedHistory);
                setError('');
            } catch (err) {
                setError('Failed to fetch betting history. Is the API server running and are you authenticated?');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchHistory();
    }, []);

    if (loading) {
        return <div className="text-center p-8">Loading betting history...</div>;
    }

    if (error) {
        return <div className="text-center p-8 text-red-500">{error}</div>;
    }

    return (
        <div className="p-8">
            <h1 className="text-4xl font-bold mb-8 text-gray-800">Betting History & Performance</h1>

            <h2 className="text-2xl font-semibold mb-4 text-gray-700">Bankroll Evolution</h2>
            <div className="bg-white rounded-lg shadow-md p-4 mb-8" style={{ height: '400px' }}>
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis domain={['auto', 'auto']} />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="bankroll_after_bet" name="Bankroll" stroke="#8884d8" activeDot={{ r: 8 }} />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <h2 className="text-2xl font-semibold mb-4 text-gray-700">Bet History</h2>
            <div className="bg-white rounded-lg shadow-md overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                        <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Stake</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Odds</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Result</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Profit</th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {history.map((bet, index) => (
                            <tr key={index}>
                                <td className="px-6 py-4 whitespace-nowrap">{bet.date}</td>
                                <td className="px-6 py-4 whitespace-nowrap">${parseFloat(bet.stake).toFixed(2)}</td>
                                <td className="px-6 py-4 whitespace-nowrap">{parseFloat(bet.odds).toFixed(2)}</td>
                                <td className={`px-6 py-4 whitespace-nowrap ${bet.result === 'win' ? 'text-green-600' : 'text-red-600'}`}>{bet.result}</td>
                                <td className="px-6 py-4 whitespace-nowrap">${bet.profit}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default History;

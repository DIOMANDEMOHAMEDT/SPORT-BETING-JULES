import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

const Dashboard = () => {
    const [bets, setBets] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchBets = async () => {
            try {
                setLoading(true);
                // Note: In a real app, the JWT token would be retrieved from storage
                // (e.g., localStorage) and included in the request headers.
                // const token = localStorage.getItem('token');
                // const response = await axios.get(`${API_URL}/get_today_bets`, { headers: { Authorization: `Bearer ${token}` } });
                const response = await axios.get(`${API_URL}/get_today_bets`); // Simplified for demo
                setBets(response.data.bets || []);
                setError('');
            } catch (err) {
                setError('Failed to fetch daily bets. Is the API server running and are you authenticated?');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchBets();
    }, []);

    if (loading) {
        return <div className="text-center p-8">Loading today's recommendations...</div>;
    }

    if (error) {
        return <div className="text-center p-8 text-red-500">{error}</div>;
    }

    return (
        <div className="p-8">
            <h1 className="text-4xl font-bold mb-8 text-gray-800">Today's Recommended Parlays</h1>
            {bets.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {bets.map((parlay, index) => (
                        <div key={index} className="bg-white rounded-lg shadow-md p-6">
                            <h2 className="text-2xl font-semibold text-gray-700 mb-4">Parlay #{index + 1}</h2>
                            <ul className="mb-4">
                                {parlay.matches.map((match, matchIndex) => (
                                    <li key={matchIndex} className="text-gray-600 text-lg mb-1">{match}</li>
                                ))}
                            </ul>
                            <div className="text-lg">
                                <p><strong>Total Odds:</strong> <span className="font-mono text-blue-600">{parlay.total_odds.toFixed(2)}</span></p>
                                <p><strong>Model Probability:</strong> <span className="font-mono text-green-600">{(parlay.total_prob * 100).toFixed(2)}%</span></p>
                                <p className="mt-4 text-xl"><strong>Recommended Stake:</strong> <span className="font-bold text-indigo-700">${parlay.stake.toFixed(2)}</span></p>
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="text-center p-8 bg-white rounded-lg shadow-md">
                    <p className="text-xl text-gray-600">No parlays recommended for today.</p>
                </div>
            )}
        </div>
    );
};

export default Dashboard;

import { useState, useRef, useEffect } from 'react';

export default function ControlPanel({ audioUrl }) {
    const audioRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [speed, setSpeed] = useState(1);

    // Keep the audio element in sync with speed changes
    useEffect(() => {
        if (audioRef.current) {
            audioRef.current.playbackRate = speed;
        }
    }, [speed]);

    // Reset state when audio URL changes
    useEffect(() => {
        setIsPlaying(false);
        if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current.currentTime = 0;
            audioRef.current.playbackRate = speed;
        }
    }, [audioUrl]);

    const play = () => {
        if (!audioRef.current || !audioUrl) return;
        audioRef.current.playbackRate = speed;
        audioRef.current.play();
        setIsPlaying(true);
    };

    const pause = () => {
        if (!audioRef.current) return;
        audioRef.current.pause();
        setIsPlaying(false);
    };

    const stop = () => {
        if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current.currentTime = 0;
        }
        setIsPlaying(false);
    };

    const changeSpeed = (s) => {
        setSpeed(s);
    };

    // Handle audio ending
    const handleEnded = () => {
        setIsPlaying(false);
    };

    // 2x2 grid of speeds - row-wise chunking
    const speedOptions = [
        [0.5, 1],
        [1.5, 2],
    ];

    return (
        <div className="flex items-center gap-3 bg-gray-800/60 backdrop-blur rounded-xl px-4 py-2 border border-gray-700">
            {audioUrl && (
                <audio
                    ref={audioRef}
                    src={audioUrl}
                    preload="auto"
                    onEnded={handleEnded}
                />
            )}

            <button
                onClick={stop}
                disabled={!audioUrl}
                className="px-3 py-1.5 bg-red-600/80 hover:bg-red-500 disabled:opacity-40 rounded-lg text-xs font-bold transition"
            >
                ⏹ Stop
            </button>
            {isPlaying
                ? <button onClick={pause} className="px-3 py-1.5 bg-yellow-600/80 hover:bg-yellow-500 rounded-lg text-xs font-bold transition">⏸ Pause</button>
                : <button onClick={play} disabled={!audioUrl} className="px-3 py-1.5 bg-green-600/80 hover:bg-green-500 disabled:opacity-40 rounded-lg text-xs font-bold transition">▶ Play</button>
            }

            <div className="flex items-center gap-2 ml-2">
                <span className="text-xs text-gray-400">Speed:</span>
                <div className="grid grid-cols-2 grid-rows-2 gap-1">
                    {speedOptions.flat().map((s, idx) => (
                        <button
                            key={s}
                            onClick={() => changeSpeed(s)}
                            className={`px-2 py-1 rounded text-xs font-mono transition ${speed === s ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-400 hover:bg-gray-600'}`}
                            style={{ minWidth: '34px' }}
                        >
                            {s}x
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
}

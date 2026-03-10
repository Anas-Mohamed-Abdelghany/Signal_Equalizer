import { useSignal } from '../core/SignalContext';

const DOMAINS = [
    { value: 'fourier',     label: '📊 Fourier (FFT)' },
    { value: 'dwt_symlet8', label: '🎸 DWT Symlet-8' },
    { value: 'dwt_db4',     label: '🗣️ DWT Daubechies-4' },
    { value: 'cwt_morlet',  label: '🐾 CWT Morlet' },
];

export default function DomainSelector() {
    const { domain, setDomain } = useSignal();

    return (
        <select
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
            className="bg-gray-800 text-white border border-gray-600 rounded-lg px-4 py-2 text-sm focus:ring-2 focus:ring-purple-500 focus:outline-none cursor-pointer"
        >
            {DOMAINS.map((d) => (
                <option key={d.value} value={d.value}>{d.label}</option>
            ))}
        </select>
    );
}

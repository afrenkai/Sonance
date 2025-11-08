'use client';

export default function LoadingSpinner({ message = 'Loading...' }: { message?: string }) {
	return (
		<div className="flex flex-col items-center justify-center gap-4 py-12">
			<div className="relative w-16 h-16">
				{/* Outer ring */}
				<div className="absolute inset-0 border-4 border-green-500/30 rounded-full"></div>

				{/* Spinning ring */}
				<div className="absolute inset-0 border-4 border-transparent border-t-green-500 rounded-full animate-spin"></div>

				{/* Inner pulse */}
				<div className="absolute inset-2 bg-green-500/20 rounded-full animate-pulse"></div>
			</div>

			<p className="text-sm text-zinc-400">{message}</p>
		</div>
	);
}

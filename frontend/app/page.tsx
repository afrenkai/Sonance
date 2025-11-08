'use client';

import { useState, useEffect } from 'react';
import {
	spotifyAPI,
	SpotifyTrack,
	PlaylistGenerationResponse,
} from '@/lib/spotify-api';
import { getEmotionColor } from '@/lib/spotify-utils';
import EmotionSelector from '@/components/EmotionSelector';
import PlaylistDisplay from '@/components/PlaylistDisplay';
import LoadingSpinner from '@/components/LoadingSpinner';

const DEFAULT_EMOTIONS = [
	'happy',
	'sad',
	'energetic',
	'calm',
	'angry',
	'melancholic',
	'hopeful',
	'romantic',
	'anxious',
	'peaceful',
];

interface SeedSong {
	song_name: string;
	artist: string;
}

export default function Home() {
	const [emotions, setEmotions] = useState<string[]>(DEFAULT_EMOTIONS);
	const [selectedEmotion, setSelectedEmotion] = useState<string | null>(null);
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [playlistData, setPlaylistData] = useState<PlaylistGenerationResponse | null>(null);
	const [numTracks, setNumTracks] = useState(20);
	const [seedSongs, setSeedSongs] = useState<SeedSong[]>([]);
	const [currentSongName, setCurrentSongName] = useState('');
	const [currentArtist, setCurrentArtist] = useState('');
	const [generationMode, setGenerationMode] = useState<'emotion' | 'songs' | 'both'>('emotion');

	// Fetch available emotions on mount
	useEffect(() => {
		spotifyAPI
			.getEmotions()
			.then(setEmotions)
			.catch((err) => {
				console.error('Failed to fetch emotions:', err);
				// Keep default emotions
			});
	}, []);

	const addSeedSong = () => {
		if (currentSongName.trim() && currentArtist.trim()) {
			setSeedSongs([...seedSongs, { song_name: currentSongName, artist: currentArtist }]);
			setCurrentSongName('');
			setCurrentArtist('');
		}
	};

	const removeSeedSong = (index: number) => {
		setSeedSongs(seedSongs.filter((_, i) => i !== index));
	};

	const handleGeneratePlaylist = async () => {
		if (generationMode === 'emotion' && !selectedEmotion) {
			setError('Please select an emotion');
			return;
		}
		if (generationMode === 'songs' && seedSongs.length === 0) {
			setError('Please add at least one seed song');
			return;
		}
		if (generationMode === 'both' && (!selectedEmotion || seedSongs.length === 0)) {
			setError('Please select an emotion and add at least one seed song');
			return;
		}

		setIsLoading(true);
		setError(null);

		try {
			const request: any = {
				num_results: numTracks,
				include_collage: true,
			};

			if (generationMode === 'emotion' || generationMode === 'both') {
				request.emotion = [selectedEmotion!];
			}

			if (generationMode === 'songs' || generationMode === 'both') {
				request.songs = seedSongs;
			}

			const response = await spotifyAPI.generatePlaylist(request);
			setPlaylistData(response);
		} catch (err) {
			setError(err instanceof Error ? err.message : 'Failed to generate playlist');
			console.error('Error generating playlist:', err);
		} finally {
			setIsLoading(false);
		}
	};

	const gradientColors = selectedEmotion
		? getEmotionColor(selectedEmotion)
		: { primary: '#1db954', secondary: '#1ed760' };

	return (
		<div className="min-h-screen bg-gradient-to-br from-black via-zinc-900 to-black">
			{/* Hero Section */}
			<div
				className="relative overflow-hidden"
				style={{
					background: `linear-gradient(180deg, ${gradientColors.primary}20 0%, transparent 100%)`,
				}}
			>
				<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
					{/* Header */}
					<div className="text-center mb-12">
						<h1 className="text-5xl md:text-7xl font-bold text-white mb-4">
							🎵 EmoRec
						</h1>
						<p className="text-xl text-zinc-400 max-w-2xl mx-auto">
							Discover music that matches your emotions. Powered by AI-driven mood
							analysis and Spotify.
						</p>
					</div>

					{/* Generation Mode Tabs */}
					<div className="flex justify-center mb-8">
						<div className="inline-flex rounded-lg bg-white/10 p-1">
							<button
								onClick={() => setGenerationMode('emotion')}
								className={`px-6 py-2 rounded-md transition-all ${
									generationMode === 'emotion'
										? 'bg-green-500 text-black font-semibold'
										: 'text-white hover:text-green-400'
								}`}
							>
								😊 By Emotion
							</button>
							<button
								onClick={() => setGenerationMode('songs')}
								className={`px-6 py-2 rounded-md transition-all ${
									generationMode === 'songs'
										? 'bg-green-500 text-black font-semibold'
										: 'text-white hover:text-green-400'
								}`}
							>
								🎵 By Songs
							</button>
							<button
								onClick={() => setGenerationMode('both')}
								className={`px-6 py-2 rounded-md transition-all ${
									generationMode === 'both'
										? 'bg-green-500 text-black font-semibold'
										: 'text-white hover:text-green-400'
								}`}
							>
								🎯 Both
							</button>
						</div>
					</div>

					{/* Emotion Selector */}
					{(generationMode === 'emotion' || generationMode === 'both') && (
						<div className="mb-8">
							<EmotionSelector
								emotions={emotions}
								selectedEmotion={selectedEmotion}
								onSelect={setSelectedEmotion}
							/>
						</div>
					)}

					{/* Seed Songs Input */}
					{(generationMode === 'songs' || generationMode === 'both') && (
						<div className="mb-8 max-w-3xl mx-auto">
							<h3 className="text-lg font-semibold text-white mb-4 text-center">
								Add Seed Songs
							</h3>
							<div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
								{/* Input fields */}
								<div className="flex flex-col sm:flex-row gap-3 mb-4">
									<input
										type="text"
										placeholder="Song name"
										value={currentSongName}
										onChange={(e) => setCurrentSongName(e.target.value)}
										onKeyPress={(e) => e.key === 'Enter' && addSeedSong()}
										className="flex-1 px-4 py-3 rounded-lg bg-white/10 text-white border border-white/20 focus:outline-none focus:ring-2 focus:ring-green-500 placeholder-zinc-500"
									/>
									<input
										type="text"
										placeholder="Artist"
										value={currentArtist}
										onChange={(e) => setCurrentArtist(e.target.value)}
										onKeyPress={(e) => e.key === 'Enter' && addSeedSong()}
										className="flex-1 px-4 py-3 rounded-lg bg-white/10 text-white border border-white/20 focus:outline-none focus:ring-2 focus:ring-green-500 placeholder-zinc-500"
									/>
									<button
										onClick={addSeedSong}
										disabled={!currentSongName.trim() || !currentArtist.trim()}
										className="px-6 py-3 rounded-lg bg-green-500 text-black font-semibold hover:bg-green-400 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-all"
									>
										Add Song
									</button>
								</div>

								{/* Seed songs list */}
								{seedSongs.length > 0 && (
									<div className="space-y-2">
										{seedSongs.map((song, index) => (
											<div
												key={index}
												className="flex items-center justify-between bg-white/5 rounded-lg p-3"
											>
												<div className="flex-1">
													<div className="text-white font-medium">{song.song_name}</div>
													<div className="text-sm text-zinc-400">{song.artist}</div>
												</div>
												<button
													onClick={() => removeSeedSong(index)}
													className="px-3 py-1 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
												>
													Remove
												</button>
											</div>
										))}
									</div>
								)}

								{seedSongs.length === 0 && (
									<div className="text-center text-zinc-500 py-4">
										No seed songs added yet. Add some to get started!
									</div>
								)}
							</div>
						</div>
					)}

					{/* Controls */}
					<div className="flex flex-col sm:flex-row items-center justify-center gap-4">
						<div className="flex items-center gap-3">
							<label htmlFor="numTracks" className="text-sm text-zinc-400">
								Number of tracks:
							</label>
							<select
								id="numTracks"
								value={numTracks}
								onChange={(e) => setNumTracks(Number(e.target.value))}
								className="px-4 py-2 rounded-lg bg-white/10 text-white border border-white/20 focus:outline-none focus:ring-2 focus:ring-green-500"
							>
								<option value={10}>10</option>
								<option value={20}>20</option>
								<option value={30}>30</option>
								<option value={50}>50</option>
							</select>
						</div>

						<button
							onClick={handleGeneratePlaylist}
							disabled={
								isLoading ||
								(generationMode === 'emotion' && !selectedEmotion) ||
								(generationMode === 'songs' && seedSongs.length === 0) ||
								(generationMode === 'both' && (!selectedEmotion || seedSongs.length === 0))
							}
							className="px-8 py-3 rounded-full bg-green-500 text-black font-semibold hover:bg-green-400 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-all transform hover:scale-105 active:scale-95"
						>
							{isLoading ? 'Generating...' : '🎵 Generate Playlist'}
						</button>
					</div>

					{/* Error Message */}
					{error && (
						<div className="mt-6 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-center max-w-2xl mx-auto">
							{error}
						</div>
					)}
				</div>
			</div>

			{/* Results Section */}
			<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
				{isLoading && (
					<LoadingSpinner message="Creating your perfect playlist..." />
				)}

				{!isLoading && playlistData && (
					<div className="space-y-8">

						{/* Playlist */}
						<div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
							<PlaylistDisplay
								tracks={playlistData.playlist}
								emotion={selectedEmotion || undefined}
								title={`Your ${
									selectedEmotion 
										? selectedEmotion.charAt(0).toUpperCase() + selectedEmotion.slice(1) 
										: 'Custom'
								} Playlist`}
								onPlayTrack={(track) => {
									if (track.preview_url) {
										const audio = new Audio(track.preview_url);
										audio.play();
									} else if (track.external_url) {
										window.open(track.external_url, '_blank');
									}
								}}
							/>
						</div>

						{/* Generation Info */}
						<div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
							<h3 className="text-lg font-semibold text-white mb-4">
								Generation Info
							</h3>
							<div className="grid grid-cols-1 md:grid-cols-3 gap-4">
								{generationMode && (
									<div className="flex flex-col">
										<span className="text-sm text-zinc-400 mb-1">Mode</span>
										<span className="text-white font-medium capitalize">{generationMode}</span>
									</div>
								)}
								{selectedEmotion && (
									<div className="flex flex-col">
										<span className="text-sm text-zinc-400 mb-1">Emotion</span>
										<span className="text-white font-medium capitalize">{selectedEmotion}</span>
									</div>
								)}
								{seedSongs.length > 0 && (
									<div className="flex flex-col">
										<span className="text-sm text-zinc-400 mb-1">Seed Songs</span>
										<span className="text-white font-medium">{seedSongs.length} tracks</span>
									</div>
								)}
							</div>
							{seedSongs.length > 0 && (
								<div className="mt-4">
									<div className="text-sm text-zinc-400 mb-2">Based on:</div>
									<div className="flex flex-wrap gap-2">
										{seedSongs.map((song, i) => (
											<span
												key={i}
												className="px-3 py-1 rounded-full bg-white/10 text-white text-sm"
											>
												{song.song_name} - {song.artist}
											</span>
										))}
									</div>
								</div>
							)}
						</div>

						{/* Audio Features Summary */}
						{playlistData.emotion_features && (
							<div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
								<h3 className="text-lg font-semibold text-white mb-4">
									Target Audio Features
								</h3>
								<div className="grid grid-cols-2 md:grid-cols-4 gap-4">
									{Object.entries(playlistData.emotion_features).map(
										([feature, value]) => (
											<div key={feature} className="flex flex-col">
												<span className="text-sm text-zinc-400 capitalize mb-1">
													{feature}
												</span>
												<div className="flex items-center gap-2">
													<div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
														<div
															className="h-full bg-green-500 rounded-full transition-all"
															style={{
																width: `${typeof value === 'number' ? value * 100 : 50}%`,
															}}
														/>
													</div>
													<span className="text-xs text-white font-mono">
														{typeof value === 'number'
															? (value * 100).toFixed(0)
															: value}
														%
													</span>
												</div>
											</div>
										)
									)}
								</div>
							</div>
						)}
					</div>
				)}

				{/* Empty State */}
				{!isLoading && !playlistData && !error && (
					<div className="text-center py-16">
						<div className="text-6xl mb-4">🎧</div>
						<h3 className="text-2xl font-semibold text-white mb-2">
							Ready to discover your soundtrack?
						</h3>
						<p className="text-zinc-400">
							Select an emotion above and generate your personalized playlist
						</p>
					</div>
				)}
			</div>

			{/* Footer */}
			<footer className="border-t border-white/10 mt-16">
				<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
					<div className="flex flex-col md:flex-row items-center justify-between gap-4">
						<p className="text-sm text-zinc-500">
							© 2025 EmoRec • Emotion-based music recommendation
						</p>
						<div className="flex gap-6 text-sm text-zinc-400">
							<a href="#" className="hover:text-white transition-colors">
								About
							</a>
							<a href="#" className="hover:text-white transition-colors">
								API Docs
							</a>
							<a href="#" className="hover:text-white transition-colors">
								GitHub
							</a>
						</div>
					</div>
				</div>
			</footer>
		</div>
	);
}

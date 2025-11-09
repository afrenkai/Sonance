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
import SongAutocomplete from '@/components/SongAutocomplete';
import ArtistAutocomplete from '@/components/ArtistAutocomplete';

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
	spotify_id?: string;
}

interface SeedArtist {
	artist_name: string;
	spotify_id?: string;
}

export default function Home() {
	const [emotions, setEmotions] = useState<string[]>(DEFAULT_EMOTIONS);
	const [selectedEmotion, setSelectedEmotion] = useState<string | null>(null);
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [playlistData, setPlaylistData] = useState<PlaylistGenerationResponse | null>(null);
	const [numTracks, setNumTracks] = useState(20);
	const [seedSongs, setSeedSongs] = useState<SeedSong[]>([]);
	const [seedArtists, setSeedArtists] = useState<SeedArtist[]>([]);
	const [generationMode, setGenerationMode] = useState<'emotion' | 'songs' | 'artists' | 'both'>('emotion');
  const [audioURL, setAudioURL] = useState<string | null>(null);
  
	useEffect(() => {
		spotifyAPI
			.getEmotions()
			.then(setEmotions)
			.catch((err) => {
				console.error('Failed to fetch emotions:', err);
			});
	}, []);

	const addSeedSong = (song: SeedSong) => {
		const isDuplicate = seedSongs.some(
			s => s.song_name === song.song_name && s.artist === song.artist
		);
		if (!isDuplicate) {
			setSeedSongs([...seedSongs, song]);
		}
	};

	const removeSeedSong = (index: number) => {
		setSeedSongs(seedSongs.filter((_, i) => i !== index));
	};

	const addSeedArtist = (artist: SeedArtist) => {
		const isDuplicate = seedArtists.some(
			a => a.artist_name === artist.artist_name
		);
		if (!isDuplicate) {
			setSeedArtists([...seedArtists, artist]);
		}
	};

	const removeSeedArtist = (index: number) => {
		setSeedArtists(seedArtists.filter((_, i) => i !== index));
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
		if (generationMode === 'artists' && seedArtists.length === 0) {
			setError('Please add at least one artist');
			return;
		}
		if (generationMode === 'both' && (!selectedEmotion || (seedSongs.length === 0 && seedArtists.length === 0))) {
			setError('Please select an emotion and add at least one seed song or artist');
			return;
		}

		setIsLoading(true);
		setError(null);

		try {
			const request: any = {
				num_results: numTracks,
				enrich_with_lyrics: false,
			};

			if (generationMode === 'emotion' || generationMode === 'both') {
				request.emotion = [selectedEmotion!];
			}

			if (generationMode === 'songs' || generationMode === 'both') {
				if (seedSongs.length > 0) {
					request.songs = seedSongs;
				}
			}

			if (generationMode === 'artists' || generationMode === 'both') {
				if (seedArtists.length > 0) {
					request.artists = seedArtists;
				}
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
			<div
				className="relative overflow-hidden"
				style={{
					background: `linear-gradient(180deg, ${gradientColors.primary}20 0%, transparent 100%)`,
				}}
			>
				<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
					<div className="text-center mb-12">
						<h1 className="text-5xl md:text-7xl font-bold text-white mb-4">
							EmoRec
						</h1>
						<p className="text-xl text-zinc-400 max-w-2xl mx-auto">
							Discover music that matches your emotions. Powered by AI semantic
							understanding of mood and context.
						</p>
						<p className="text-sm text-zinc-500 mt-2 max-w-2xl mx-auto">
							Uses semantic embeddings to match emotional vibes, not just genre or artist similarity
						</p>
					</div>

					<div className="flex justify-center mb-8">
						<div className="inline-flex rounded-lg bg-white/10 p-1">
							<button
								onClick={() => setGenerationMode('emotion')}
								className={`px-6 py-2 rounded-md transition-all ${generationMode === 'emotion'
									? 'bg-green-500 text-black font-semibold'
									: 'text-white hover:text-green-400'
									}`}
							>
								😊 By Emotion
							</button>
							<button
								onClick={() => setGenerationMode('songs')}
								className={`px-6 py-2 rounded-md transition-all ${generationMode === 'songs'
									? 'bg-green-500 text-black font-semibold'
									: 'text-white hover:text-green-400'
									}`}
							>
								🎵 By Songs
							</button>
							<button
								onClick={() => setGenerationMode('artists')}
								className={`px-6 py-2 rounded-md transition-all ${generationMode === 'artists'
									? 'bg-green-500 text-black font-semibold'
									: 'text-white hover:text-green-400'
									}`}
							>
								🎤 By Artists
							</button>
							<button
								onClick={() => setGenerationMode('both')}
								className={`px-6 py-2 rounded-md transition-all ${generationMode === 'both'
									? 'bg-green-500 text-black font-semibold'
									: 'text-white hover:text-green-400'
									}`}
							>
								🎯 Combined
							</button>
						</div>
					</div>

					{(generationMode === 'emotion' || generationMode === 'both') && (
						<div className="mb-8">
							<EmotionSelector
								emotions={emotions}
								selectedEmotion={selectedEmotion}
								onSelect={setSelectedEmotion}
							/>
						</div>
					)}

					{(generationMode === 'songs' || generationMode === 'both') && (
						<div className="mb-8 max-w-3xl mx-auto">
							<h3 className="text-lg font-semibold text-white mb-2 text-center">
								Add Seed Songs
							</h3>
							<p className="text-sm text-zinc-400 mb-4 text-center">
								🎭 We'll find songs with similar emotional vibe and mood using AI embeddings
							</p>
							<div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
								<div className="mb-4">
									<SongAutocomplete
										onSelectSong={addSeedSong}
										placeholder="Search for a song to add..."
									/>
								</div>

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
										No seed songs added yet. Search and select songs to get started!
									</div>
								)}
							</div>
						</div>
					)}

					{(generationMode === 'artists' || generationMode === 'both') && (
						<div className="mb-8 max-w-3xl mx-auto">
							<h3 className="text-lg font-semibold text-white mb-2 text-center">
								Add Seed Artists
							</h3>
							<p className="text-sm text-zinc-400 mb-4 text-center">
								🎤 We'll use their style and top tracks to find similar music
							</p>
							<div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
								<div className="mb-4">
									<ArtistAutocomplete
										onSelectArtist={addSeedArtist}
										placeholder="Search for an artist to add..."
									/>
								</div>

								{seedArtists.length > 0 && (
									<div className="space-y-2">
										{seedArtists.map((artist, index) => (
											<div
												key={index}
												className="flex items-center justify-between bg-white/5 rounded-lg p-3"
											>
												<div className="flex-1">
													<div className="text-white font-medium">{artist.artist_name}</div>
												</div>
												<button
													onClick={() => removeSeedArtist(index)}
													className="px-3 py-1 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
												>
													Remove
												</button>
											</div>
										))}
									</div>
								)}

								{seedArtists.length === 0 && (
									<div className="text-center text-zinc-500 py-4">
										No seed artists added yet. Search and select artists to get started!
									</div>
								)}
							</div>
						</div>
					)}

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
								(generationMode === 'artists' && seedArtists.length === 0) ||
								(generationMode === 'both' && (!selectedEmotion || (seedSongs.length === 0 && seedArtists.length === 0)))
							}
							className="px-8 py-3 rounded-full bg-green-500 text-black font-semibold hover:bg-green-400 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-all transform hover:scale-105 active:scale-95"
						>
							{isLoading ? 'Generating...' : '🎵 Generate Playlist'}
						</button>
					</div>

					{error && (
						<div className="mt-6 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-center max-w-2xl mx-auto">
							{error}
						</div>
					)}
				</div>
			</div>

			<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
				{isLoading && (
					<LoadingSpinner message="Finding songs with matching emotional vibes..." />
				)}

				{!isLoading && playlistData && (
					<div className="space-y-8">
            <audio src={audioURL || undefined} autoPlay/>
						<div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
							<PlaylistDisplay
								tracks={playlistData.playlist}
								emotion={selectedEmotion || undefined}
								title={`Your ${selectedEmotion
									? selectedEmotion.charAt(0).toUpperCase() + selectedEmotion.slice(1)
									: 'Custom'
									} Playlist`}
								onPlayTrack={(track) => {
                  const base = process.env.NEXT_PUBLIC_API_URL;
                  const url =
                    base +
                    "/get-audio?" +
                    `song_title=${encodeURIComponent(track.name)}` +
                    "&" +
                    `artist_name=${encodeURIComponent(track.artist)}`; 
                  setAudioURL(url);
								}}
							/>
						</div>
					</div>
				)}

				{!isLoading && !playlistData && !error && (
					<div className="text-center py-16">
						<div className="text-6xl mb-4">🎧</div>
						<h3 className="text-2xl font-semibold text-white mb-2">
							Ready to discover your soundtrack?
						</h3>
						<p className="text-zinc-400 mb-4">
							{generationMode === 'emotion' && 'Select emotion(s) above and generate your mood-based playlist'}
							{generationMode === 'songs' && 'Add seed songs - we\'ll find music with similar emotional vibes'}
							{generationMode === 'artists' && 'Add artists - we\'ll explore their style and find similar music'}
							{generationMode === 'both' && 'Select emotion(s) and add songs/artists for precise mood matching'}
						</p>
						<p className="text-xs text-zinc-500 max-w-md mx-auto">
							💡 Our AI uses semantic embeddings to understand emotional context,
							finding songs that match the vibe across different genres and artists
						</p>
					</div>
				)}
			</div>

			<footer className="border-t border-white/10 mt-16">
				<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
					<div className="flex flex-col md:flex-row items-center justify-between gap-4">
						<p className="text-sm text-zinc-500">
							© 2025 EmoRec Team • Emotion-based music recommendation
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

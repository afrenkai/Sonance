/**
 * Spotify-themed track card component
 */

'use client';

import { SpotifyTrack } from '@/lib/spotify-api';
import { formatSimilarityScore, getTrackPlaceholder, formatDuration } from '@/lib/spotify-utils';

interface TrackCardProps {
  track: SpotifyTrack;
  index: number;
  emotion?: string;
  onPlay?: (track: SpotifyTrack) => void;
  isActive?: boolean;
  isPlaying?: boolean;
}
export default function TrackCard({ track, index, emotion, onPlay, isActive, isPlaying }: TrackCardProps) {
  const handleOpenSpotify = () => {
    if (track.external_url) {
      window.open(track.external_url, '_blank');
    }
  };

  return (
    <div className={`group flex items-center gap-4 rounded-lg p-3 transition-colors ${isActive ? 'bg-white/5 ring-1 ring-green-400' : 'hover:bg-white/5'}`}>
      {/* Track number */}
      <div className="w-8 text-center text-zinc-400 group-hover:text-green-500">
        <button
          onClick={() => onPlay?.(track)}
          className="w-full text-sm"
          title={isActive && isPlaying ? 'Pause' : 'Play'}
        >
          {isActive && isPlaying ? '⏸' : '▶'}
        </button>
      </div>

      {/* Album art or placeholder */}
      {track.album_image ? (
        <img
          src={track.album_image}
          alt={track.album || 'Album art'}
          className="w-12 h-12 rounded object-cover"
        />
      ) : (
        <div
          className="w-12 h-12 rounded flex items-center justify-center text-white text-xs font-bold"
          style={{ background: getTrackPlaceholder(emotion) }}
        >
          🎵
        </div>
      )}

      {/* Track info */}
      <div className="flex-1 min-w-0">
        <div className="font-medium text-white truncate hover:underline cursor-pointer" onClick={handleOpenSpotify}>
          {track.name}
        </div>
        <div className="text-sm text-zinc-400 truncate">{track.artist}</div>
      </div>

      {/* Album */}
      {track.album && (
        <div className="hidden md:block text-sm text-zinc-400 truncate max-w-[200px]">
          {track.album}
        </div>
      )}

      {/* Duration */}
      {track.duration_ms && (
        <div className="hidden sm:block text-sm text-zinc-500">
          {formatDuration(track.duration_ms)}
        </div>
      )}

      {/* Similarity score */}
      <div className="text-sm text-zinc-500">
        {formatSimilarityScore(track.similarity_score)}
      </div>

      {/* Audio features indicator (optional) */}
      {((track as any).audio_features) && (
        <div className="hidden lg:flex gap-1">
          {((track as any).audio_features.energy > 0.7) && (
            <span className="text-xs px-2 py-1 rounded-full bg-red-500/20 text-red-400">
              High Energy
            </span>
          )}
          {((track as any).audio_features.valence > 0.7) && (
            <span className="text-xs px-2 py-1 rounded-full bg-yellow-500/20 text-yellow-400">
              Upbeat
            </span>
          )}
          {((track as any).audio_features.danceability > 0.7) && (
            <span className="text-xs px-2 py-1 rounded-full bg-purple-500/20 text-purple-400">
              Danceable
            </span>
          )}
        </div>
      )}

      {/* Spotify link */}
      {track.external_url && (
        <button
          onClick={handleOpenSpotify}
          className="opacity-0 group-hover:opacity-100 transition-opacity px-3 py-1.5 rounded-full bg-green-500 text-black text-sm font-medium hover:bg-green-400"
          title="Open in Spotify"
        >
          <span className="hidden xl:inline">Open in Spotify</span>
          <span className="xl:hidden">🎵</span>
        </button>
      )}

      {/* Preview audio button */}
      {track.preview_url && (
        <button
          onClick={() => {
            const audio = new Audio(track.preview_url);
            audio.play();
          }}
          className="opacity-0 group-hover:opacity-100 transition-opacity px-3 py-1.5 rounded-full bg-white/10 text-white text-sm font-medium hover:bg-white/20"
          title="Preview"
        >
          ▶️
        </button>
      )}
    </div>
  );
}

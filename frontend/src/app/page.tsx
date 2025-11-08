"use client";

import { useState } from "react";
import PlaylistGenerator from "@/components/PlaylistGenerator";
import PlaylistDisplay from "@/components/PlaylistDisplay";
import MoodCollage from "@/components/MoodCollage";
import { PlaylistResponse } from "@/types";

export default function Home() {
  const [playlistData, setPlaylistData] = useState<PlaylistResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePlaylistGenerated = (data: PlaylistResponse) => {
    setPlaylistData(data);
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen flex flex-col" style={{ backgroundColor: 'var(--background)' }}>
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-black/50 border-b" style={{ borderColor: 'var(--border)' }}>
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ backgroundColor: 'var(--accent)' }}>
              <svg className="w-6 h-6 text-black" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
              </svg>
            </div>
            <h1 className="text-2xl font-bold">EmoRec</h1>
          </div>
          <div className="text-sm" style={{ color: 'var(--foreground-secondary)' }}>
            Emotion-Based Music Discovery
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto px-6 py-8">
          {/* Hero Section */}
          {!playlistData && (
            <div className="mb-12 text-center py-16">
              <h2 className="text-5xl font-bold mb-4">
                Find music that matches your{" "}
                <span style={{ color: 'var(--accent)' }}>mood</span>
              </h2>
              <p className="text-xl" style={{ color: 'var(--foreground-secondary)' }}>
                Discover personalized playlists based on your emotions and favorite songs
              </p>
            </div>
          )}

          {/* Generator Section */}
          <div className="mb-8">
            <PlaylistGenerator
              onPlaylistGenerated={handlePlaylistGenerated}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
            />
          </div>

          {/* Results Section */}
          {playlistData && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Playlist */}
              <div className="lg:col-span-2">
                <PlaylistDisplay playlist={playlistData.playlist} />
              </div>

              {/* Mood Collage */}
              {playlistData.mood_collage && (
                <div className="lg:col-span-1">
                  <MoodCollage collage={playlistData.mood_collage} />
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t py-6" style={{ borderColor: 'var(--border)', backgroundColor: 'var(--background-elevated)' }}>
        <div className="max-w-7xl mx-auto px-6 text-center text-sm" style={{ color: 'var(--foreground-secondary)' }}>
          <p>Powered by emotion-based embeddings and music recommendation</p>
        </div>
      </footer>
    </div>
  );
}

'use client';

import { useState } from 'react';
import { getEmotionColor, getEmotionEmoji } from '@/lib/spotify-utils';

interface EmotionSelectorProps {
	emotions: string[];
	selectedEmotion: string | null;
	onSelect: (emotion: string) => void;
}

export default function EmotionSelector({
	emotions,
	selectedEmotion,
	onSelect,
}: EmotionSelectorProps) {
	const [inputValue, setInputValue] = useState('');
	const [showSuggestions, setShowSuggestions] = useState(false);

	const filteredEmotions = emotions.filter((emotion) =>
		emotion.toLowerCase().includes(inputValue.toLowerCase())
	);

	const handleInputChange = (value: string) => {
		setInputValue(value);
		setShowSuggestions(value.length > 0);
	};

	const handleSelectEmotion = (emotion: string) => {
		onSelect(emotion);
		setInputValue(emotion);
		setShowSuggestions(false);
	};

	const handleSubmit = () => {
		if (!inputValue.trim()) return;

		const exactMatch = emotions.find(
			(emotion) => emotion.toLowerCase() === inputValue.toLowerCase()
		);
		if (exactMatch) {
			handleSelectEmotion(exactMatch);
		} else if (filteredEmotions.length > 0) {
			handleSelectEmotion(filteredEmotions[0]);
		} else {
			// Allow custom emotions
			onSelect(inputValue.trim().toLowerCase());
			setShowSuggestions(false);
		}
	};

	const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
		if (e.key === 'Enter') {
			handleSubmit();
		}
	};

	const emoji = selectedEmotion ? getEmotionEmoji(selectedEmotion) : '😊';

	return (
		<div className="w-full max-w-3xl mx-auto">
			<h3 className="text-lg font-semibold text-white mb-4 text-center">
				How are you feeling?
			</h3>
			<div className="relative">
				<div className="flex gap-3">
					<div className="relative flex-1">
						<input
							type="text"
							placeholder="Type an emotion (e.g., happy, sad, energetic) or create your own..."
							value={inputValue}
							onChange={(e) => handleInputChange(e.target.value)}
							onKeyPress={handleKeyPress}
							onFocus={() => inputValue && setShowSuggestions(true)}
							onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
							className="w-full px-4 py-3 pl-12 rounded-lg bg-white/10 text-white border border-white/20 focus:outline-none focus:ring-2 focus:ring-green-500 placeholder-zinc-500"
						/>
						<span className="absolute left-4 top-1/2 -translate-y-1/2 text-2xl">
							{emoji}
						</span>

						{/* Suggestions Dropdown */}
						{showSuggestions && (
							<div className="absolute z-10 w-full mt-2 bg-zinc-800 border border-white/20 rounded-lg shadow-xl max-h-60 overflow-y-auto">
								{filteredEmotions.length > 0 ? (
									filteredEmotions.map((emotion) => {
										const colors = getEmotionColor(emotion);
										const emotionEmoji = getEmotionEmoji(emotion);
										return (
											<button
												key={emotion}
												onClick={() => handleSelectEmotion(emotion)}
												className="w-full px-4 py-3 text-left hover:bg-white/10 transition-colors flex items-center gap-3 border-b border-white/5 last:border-b-0"
											>
												<span className="text-2xl">{emotionEmoji}</span>
												<span className="text-white capitalize font-medium">
													{emotion}
												</span>
												<div
													className="ml-auto w-3 h-3 rounded-full"
													style={{
														background: `linear-gradient(135deg, ${colors.primary}, ${colors.secondary})`,
													}}
												/>
											</button>
										);
									})
								) : (
									<button
										onClick={handleSubmit}
										className="w-full px-4 py-3 text-left hover:bg-white/10 transition-colors flex items-center gap-3"
									>
										<span className="text-2xl">✨</span>
										<div className="flex-1">
											<div className="text-white font-medium">
												Use "{inputValue}" as custom emotion
											</div>
											<div className="text-xs text-zinc-400">
												Create your own emotion
											</div>
										</div>
									</button>
								)}
							</div>
						)}
					</div>
					<button
						onClick={handleSubmit}
						disabled={!inputValue.trim()}
						className="px-6 py-3 rounded-lg bg-green-500 text-black font-semibold hover:bg-green-400 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-all whitespace-nowrap"
					>
						Set Emotion
					</button>
				</div>

				{/* Selected Emotion Display */}
				{selectedEmotion && (
					<div className="mt-4 flex items-center justify-center gap-3">
						<div
							className="px-6 py-3 rounded-full flex items-center gap-3 border-2 border-white/30"
							style={{
								background: `linear-gradient(135deg, ${getEmotionColor(selectedEmotion).primary}, ${getEmotionColor(selectedEmotion).secondary})`,
							}}
						>
							<span className="text-2xl">{getEmotionEmoji(selectedEmotion)}</span>
							<span className="text-white font-semibold capitalize text-lg">
								{selectedEmotion}
							</span>
							<button
								onClick={() => {
									onSelect('');
									setInputValue('');
								}}
								className="ml-2 text-white/80 hover:text-white transition-colors"
							>
								✕
							</button>
						</div>
					</div>
				)}
			</div>
		</div>
	);
}

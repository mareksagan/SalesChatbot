package com.saleschatbot;

import opennlp.tools.doccat.DoccatModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import static com.saleschatbot.OpenNLP.*;

public class ChatBot {
	private static final Logger logger = LoggerFactory.getLogger(ChatBot.class);

	public static void main(String[] args) throws IOException {
		// Train categorizer model to the training data we created
		DoccatModel model = trainCategorizerModel("categorizer.txt");
		// Take chat inputs from console (user) in a loop
		Scanner scanner = new Scanner(System.in);
		Map<String, String> questionAnswer = new HashMap<>();
		try (BufferedReader br = new BufferedReader(new FileReader("questions.txt"))) {
			String line;
			while ((line = br.readLine()) != null) {
				if (line.isEmpty()) break;
				String[] categoryQuestion = line.split(";");
				questionAnswer.put(categoryQuestion[0], categoryQuestion[1]);
			}
		}
		while (true) {
			// Get chat input from user
			System.out.println("You:");
			String userInput = scanner.nextLine();
			// Break users chat input into sentences using sentence detection
			String[] sentences = breakSentences(userInput, "models/sentence.bin");
			StringBuilder answer = new StringBuilder();
			boolean conversationComplete = false;

			// Loop through sentences
			for (String sentence : sentences) {
				// Separate words from each sentence using tokenizer
				String[] tokens = tokenizeSentence(sentence, "models/tokenizer.bin");
				// Tag separated words with POS tags to understand their gramatical structure
				String[] posTags = detectPOSTags(tokens, "models/pos_maxent.bin");
				// Lemmatize each word so that its easy to categorize
				String[] lemmas = lemmatizeTokens(tokens, posTags, "models/lemmatizer.bin");
				// Determine BEST category using lemmatized tokens used a mode that we trained at start
				String category = detectCategory(model, lemmas);
				// Get predefined answer from given category & add to answer
				answer.append(" ").append(questionAnswer.get(category));
				// If category conversation-complete, we will end chat conversation
				if (category.equals("conversation-complete")) conversationComplete = true;
			}

			// Print answer or quit
			System.out.println("Chatbot:" + answer);
			if (conversationComplete) {
				break;
			}
		}
	}
}

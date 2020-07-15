package com.saleschatbot;

import opennlp.tools.doccat.*;
import opennlp.tools.lemmatizer.LemmatizerME;
import opennlp.tools.lemmatizer.LemmatizerModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.*;
import opennlp.tools.util.model.ModelUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public class OpenNLP {
    private static final Logger logger = LoggerFactory.getLogger(OpenNLP.class);

    // Train categorizer model as per the category sample training data we created
    public static DoccatModel trainCategorizerModel(String path) throws IOException {
        // Custom training data with categories as per our chat requirements
        InputStreamFactory inputStreamFactory = new MarkableFileInputStreamFactory(new File(path));
        ObjectStream<String> lineStream = new PlainTextByLineStream(inputStreamFactory, StandardCharsets.UTF_8);
        ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);
        DoccatFactory factory = new DoccatFactory(new FeatureGenerator[] { new BagOfWordsFeatureGenerator() });
        TrainingParameters params = ModelUtil.createDefaultTrainingParameters();
        params.put(TrainingParameters.CUTOFF_PARAM, 0);
        // Train a model with classifications from above file
        DoccatModel model = DocumentCategorizerME.train("en", sampleStream, params, factory);
        return model;
    }

    // Detect category using given token. Use categorizer
    public static String detectCategory(DoccatModel model, String[] finalTokens) {
        // Initialize document categorizer tool
        DocumentCategorizerME categorizer = new DocumentCategorizerME(model);
        // Get best possible category
        double[] probabilitiesOfOutcomes = categorizer.categorize(finalTokens);
        String category = categorizer.getBestCategory(probabilitiesOfOutcomes);
        logger.debug("Category: " + category);
        return category;
    }

    // Break data into sentences using sentence detection
    public static String[] breakSentences(String data, String path) throws IOException {
        // Better to read file once at start of program and store model in instance variable
        try (InputStream model = new FileInputStream(path)) {
            SentenceDetectorME myCategorizer = new SentenceDetectorME(new SentenceModel(model));
            String[] sentences = myCategorizer.sentDetect(data);
            logger.debug("Sentence detection: " + String.join(" | ", sentences));
            return sentences;
        }
    }

    // Break sentence into words and punctuation marks using tokenizer
    public static String[] tokenizeSentence(String sentence, String path) throws IOException {
        try (InputStream model = new FileInputStream(path)) {
            // Initialize tokenizer tool
            TokenizerME myCategorizer = new TokenizerME(new TokenizerModel(model));
            // Tokenize sentence
            String[] tokens = myCategorizer.tokenize(sentence);
            logger.debug("Tokenizer: " + String.join(" | ", tokens));
            return tokens;
        }
    }

    // Find part-of-speech or POS tags of all tokens using POS tagger
    public static String[] detectPOSTags(String[] tokens, String path) throws IOException {
        // Better to read file once at start of program and store model in instance variable
        try (InputStream model = new FileInputStream(path)) {
            // Initialize POS tagger tool
            POSTaggerME myCategorizer = new POSTaggerME(new POSModel(model));
            // Tag sentence
            String[] posTokens = myCategorizer.tag(tokens);
            logger.debug("POS Tags: " + String.join(" | ", posTokens));
            return posTokens;
        }
    }

    // Find lemma of tokens using lemmatizer
    public static String[] lemmatizeTokens(String[] tokens, String[] posTags, String path) throws IOException {
        // Better to read file once at start of program and store model in instance variable
        try (InputStream model = new FileInputStream(path)) {
            // Tag sentence
            LemmatizerME myCategorizer = new LemmatizerME(new LemmatizerModel(model));
            String[] lemmaTokens = myCategorizer.lemmatize(tokens, posTags);
            logger.debug("Lemmatizer: " + String.join(" | ", lemmaTokens));
            return lemmaTokens;
        }
    }
}

/*
 * SentimentClassifier.h
 *
 *  Created on: Dec 25, 2009
 *      Author: Christopher L. Tang
 */

#ifndef SENTIMENTCLASSIFIER_H_
#define SENTIMENTCLASSIFIER_H_

#include <string>
#include <iostream>
#include <vector>
#include <map>
//#include <boost/unordered_map.hpp>

using namespace std;

struct CDecision
// data structure for storing classification decisions
{
	CDecision();

	int decision;
	// classification decision; 1: positive; 0: neutral; -1: negative

	int score;
	// confidence score; Range TBD

	int negative;
	int neutral;
	int positive;
	// individual scores for each classification

	string content;
	// normalized content produced by Classifier

	vector<string> features;
	// features contributing classification decision
};

struct FeatureScores
// data structure for storing feature scores
{
	FeatureScores();

	int negative;
	int neutral;
	int positive;
	// feature score per class

	int relevance;
	// relevance score of feature
};

//typedef boost::unordered_map<string,int> StopwordsTable;
//typedef boost::unordered_map<string,FeatureScores> FeaturesTable;

typedef map<string,int> StopwordsTable;
typedef map<string,FeatureScores> FeaturesTable;

class SentimentClassifier {
public:
	SentimentClassifier ( const string& feature_file,
						  const string& stopword_file );
	bool Inited () const;
	bool Classify ( const string& input, CDecision& cd );
	bool Classify ( const string& title, const string& body,
				    const string& url, CDecision& cd );

	void setRelevanceCutoff ( float rc );
	void setMaxFeatureSize ( unsigned int mfs );
	void setDebugLevel ( unsigned int dl );

	float getRelevanceCutoff () const;
	unsigned int getMaxFeatureSize () const;
	unsigned int getDebugLevel () const;
	string getErrorMsg () const;

private:
	float RelevanceCutoff;
	unsigned int MaxFeatureSize;
	unsigned int DebugLevel;
	string error_msg;

	int TitleWeight;
	int BodyWeight;
	int URLWeight;

	bool isInited;
	bool readFeatures ( const string& features_file );
	bool readStopwords ( const string& stopwords_file );
	bool parseFeature ( string& phrase, string& entry );
	bool scaleRawValue ( float raw_value, int& scaled_value );
	bool normalizeContent ( const string& content, string& ncontent );
	bool normalizeUrl ( const string& content, string& ncontent );
	bool classifyGreedy ( int weight, string& ncontent, CDecision& cd );
	float getRawValue ( int scaled_value );

	FeaturesTable features;
	StopwordsTable stopwords;

	static const float FeatureScoreScale = 1000.f;
};

#endif /* SENTIMENTCLASSIFIER_H_ */

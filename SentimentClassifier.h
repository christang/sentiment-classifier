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

	int raw_score;
	// decision score; open range

	int confidence;
	// decision confidence; normalized score per feature [ -1000, 1000 ]

	string content;
	// normalized content produced by Classifier

	vector<string> features;
	// features contributing classification decision
};

struct FeatureScores
// data structure for storing feature scores
{
	FeatureScores();

	int score;
	// composite feature score

	int relevance;
	// relevance score of feature
};

//typedef boost::unordered_map<string,int> StopwordsTable;
//typedef boost::unordered_map<string,FeatureScores> FeaturesTable;

typedef map<string,int> StopwordsTable;
typedef map<string,int> FeaturesCount;
typedef map<string,FeatureScores> FeaturesTable;

class SentimentClassifier {
public:
	enum ContentType
	{
		Regular, Twitter
	};

	static const float FeatureScoreScale = 288.f; // = 200/ln(2)

	SentimentClassifier ( const string& feature_file,
						  const string& stopword_file );
	bool Inited () const;
	bool Classify ( const string& input, CDecision& cd, int type=Regular );
	bool Classify ( const string& title, const string& body,
				    const string& url, CDecision& cd, int type=Regular );

	void setUseQuestionMarks ( bool qm );
	void setUseExclamationPoints ( bool ep );
	void setRelevanceCutoff ( float rc );
	void setNeutralCutoff ( float nc );
	void setMaxFeatureSize ( unsigned int mfs );
	void setDebugLevel ( unsigned int dl );
	void setPreNormalized ( bool pn );

	bool getUseQuestionMarks () const;
	bool getUseExclamationPoints () const;
	float getRelevanceCutoff () const;
	float getNeutralCutoff () const;
	unsigned int getMaxFeatureSize () const;
	unsigned int getDebugLevel () const;
	bool getPreNormalized () const;
	string getErrorMsg () const;

private:
	bool UseQuestionMarks;
	bool UseExclamationPoints;
	bool UseEmoticons;
	float RelevanceCutoff;
	float NeutralCutoff;
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
	bool normalizeContent ( const string& content, string& ncontent );
	bool normalizeTweet ( const string& content, string& ncontent );
	bool normalizeUrl ( const string& content, string& ncontent );
	bool classifyGreedy ( int weight, string& ncontent, CDecision& cd,
			int ct, float rc, float nc );
	bool classifySentences ( int weight, const string& ucontent,
			CDecision& cd, int ct, float rc, float nc );
	void classifyExclamationPoints ( int weight,
			const string& ucontent, CDecision& cd );
	void classifyQuestionMarks ( int weight,
			const string& ucontent, CDecision& cd );
	void classifyTweetQuestionMarks ( int weight,
			const string& ucontent, CDecision& cd );

	FeaturesTable features;
	StopwordsTable stopwords;

	bool pre_normalized;

};

#endif /* SENTIMENTCLASSIFIER_H_ */

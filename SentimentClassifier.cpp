/*
 * SentimentClassifier.cpp
 *
 *  Created on: Dec 25, 2009
 *      Author: Christopher L. Tang
 */

#include "SentimentClassifier.h"

#include <math.h>
#include <fstream>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <boost/xpressive/xpressive.hpp>

using namespace boost::xpressive;

bool SentimentClassifier::classifySentences ( int weight,
		const string& ucontent, CDecision& cd )
{
	string content ( ucontent );

	sregex urlx = sregex::compile( "(http:[\\/\\w\\d\\.\\=\\&\\?]+)" );
	content = regex_replace ( content, urlx, " " );

	vector<string> sentences;
	boost::split ( sentences, content, boost::is_any_of ( ";?!" ) );

	for ( vector<string>::iterator sentence = sentences.begin();
			sentence != sentences.end(); sentence++ ) {
		CDecision cd_s;
		string nsentence;

		if ( normalizeContent ( *sentence, nsentence ) )
			classifyGreedy ( weight, nsentence, cd_s );
		else return false;

		cd.content += cd_s.content + "; ";
		cd.raw_score += cd_s.raw_score;
		cd.confidence += cd_s.confidence;
		cd.features.insert(cd.features.end(),
						   cd_s.features.begin(),cd_s.features.end());
	}

	if ( cd.features.size() == 0 ) {
		cd.confidence = -1;
		error_msg = "no decision could be reached";
	} else {
		int min_sentiment = int ( FeatureScoreScale * NeutralCutoff );

		// confidence is average relevance normalized over observed features
		cd.confidence /= int ( sentences.size() );

		// decision is based on sign of score
		if ( cd.raw_score )
			cd.decision = ( cd.raw_score < 0 ) ? -1 : 1;

		// decision is neutral if score doesn't exceed threshold
		if ( abs ( cd.raw_score ) < min_sentiment )
			cd.decision = 0;

		// decision is neutral if confidence is low
		// if ( cd.confidence < min_sentiment ) cd.decision = 0;
	}

	return ( cd.confidence >= 0 );
}

bool SentimentClassifier::classifyGreedy ( int weight,
		string& content, CDecision& cd )
{

	try {
		int cutoff = int ( FeatureScoreScale * RelevanceCutoff );
		int min_sentiment = int ( FeatureScoreScale * NeutralCutoff );

		cd.content = content;

		vector<string> tokens;
		boost::split ( tokens, content, boost::is_any_of ( " " ) );

		if ( DebugLevel > 2 )
			cout << "Content? " << cd.content << endl;

		FeaturesCount fc;

		for ( unsigned int i=0; i < tokens.size(); ++i ) {
			string test_feature = "";
			unsigned int s = MaxFeatureSize;

			for ( ; s > 0; --s ) {
				stringstream ss;
				unsigned int t = i+s;
				if ( t <= tokens.size() ) {
					ss << tokens[i];
					for ( unsigned int u = i+1 ; u < t ; ++u )
						ss << " " << tokens[u];
					if ( DebugLevel > 2 )
						cout << "Feature? " << ss.str();

					if ( features.find ( ss.str() ) != features.end() ) {
						if ( DebugLevel > 2 )
							cout << "; YES rc = " <<
								features[ ss.str() ].relevance;
						if ( features[ ss.str() ].relevance > cutoff ) {
							if ( DebugLevel > 2 )
								cout << "; PASSES cutoff (" <<
									cutoff << ")" << endl;
							test_feature = ss.str();
							break;
						}
					} else {
						if ( DebugLevel > 2 )
							cout << "; NO";
					}
					if ( DebugLevel > 2 )
						cout << endl;
				}
			}

			if ( test_feature != "" ) {
				if ( fc.find( test_feature ) == fc.end() )
					fc[test_feature] = 0;
				fc[test_feature] ++;

				if ( DebugLevel > 1 ) cout << test_feature <<
						" (" << features[ test_feature ].score << ")"
						<< endl;

				i += s-1;
			}
		}

		for ( FeaturesCount::const_iterator it = fc.begin();
				it != fc.end(); it++ ) {

			FeatureScores* fs = &features[ it->first ];

			float feature_weight =
				( 1.f + log ( float ( it->second ) ) / log ( 2.f ) );

			int feature_score =
				int ( feature_weight * float ( fs->score ) );

			cd.raw_score += weight * feature_score;

			cd.confidence += fs->score;
			//cd.confidence += fs->relevance;

			if ( DebugLevel > 0 ) {
				stringstream ss;
				ss << it->first << " *";
				ss << it->second << " = ";
				ss << feature_score;
				cd.features.push_back( ss.str() );
			} else {
				cd.features.push_back( it->first );
			}

		}

		if ( cd.features.size() == 0 ) {
			cd.confidence = -1;
			error_msg = "no decision could be reached";
		} else {
			// confidence is average relevance normalized over observed features
			cd.confidence /= int ( cd.features.size() );
			cd.confidence = abs ( cd.confidence );

			// decision is based on sign of score
			if ( cd.raw_score )
				cd.decision = ( cd.raw_score < 0 ) ? -1 : 1;

			// decision is neutral if score doesn't exceed threshold
			if ( abs ( cd.raw_score ) < min_sentiment )
				cd.decision = 0;

			// decision is neutral if confidence is low
			// if ( cd.confidence < min_sentiment ) cd.decision = 0;
		}

	} catch (...) {
		cd.confidence = -1;
		error_msg = "error in SentimentClassifier::classifyGreedy";
	}

	return ( cd.confidence >= 0 );
}

bool SentimentClassifier::classifyQuestionMarks ( int weight,
		const string& ucontent, CDecision& cd)
{
	string feature ( ucontent );

	sregex urlx = sregex::compile( "(http:[\\/\\w\\d\\.\\=\\&\\?]+)" );
	feature = regex_replace ( feature, urlx, " " );

	float qm_ratio = float ( feature.size() );

	sregex non_qmx = sregex::compile( "[^\\?]" );
	feature = regex_replace ( feature, non_qmx, "" );

	qm_ratio = float ( feature.size() ) / qm_ratio;
	float raw_score = 0.f;
	if ( qm_ratio > 0.001f )
		raw_score = FeatureScoreScale * ( -156.f * qm_ratio - 0.3f );

	//float qm_count = float ( feature.size() );
	//float raw_score = FeatureScoreScale * -0.37f * sqrt ( qm_count );

	cd.confidence = 0;
	cd.decision = cd.raw_score < 0 ? -1 : 0;
	cd.raw_score = weight * int ( raw_score );
	if ( DebugLevel > 0 ) {
		stringstream ss;
		ss << "?: '" << feature << "' = " << int ( raw_score );
		cd.features.push_back( ss.str() );
	} else {
		if ( qm_ratio > 0.f ) cd.features.push_back( feature );
	}

	return ( cd.confidence >= 0 );
}

CDecision::CDecision ()
	: decision(0), raw_score(0), confidence(0), content(), features()
{}

FeatureScores::FeatureScores ()
	: score(0), relevance(0)
{}

SentimentClassifier::SentimentClassifier (
		const string& feature_file, const string& stopword_file)
	: UseQuestionMarks (true),
	  RelevanceCutoff (1.0f), NeutralCutoff (1.0f), MaxFeatureSize (3),
	  DebugLevel (0), error_msg (), TitleWeight (3), BodyWeight (1),
	  URLWeight (1), isInited (false), features (), stopwords ()
{
	isInited =
			readFeatures (feature_file);
//			&& readStopwords (stopword_file);
}

bool SentimentClassifier::Classify (
		const string& content, CDecision& cd)
// return true if sentiment classification is successful; return false otherwise;
{

	if ( ! classifySentences ( 1, content, cd ) )
		return false;

	if ( UseQuestionMarks ) {
		CDecision cd_qm;
		classifyQuestionMarks ( 1, content, cd_qm );

		cd.features.push_back( cd_qm.features[0] );
		cd.raw_score += cd_qm.raw_score;

		int min_sentiment = int ( FeatureScoreScale * NeutralCutoff );

		// decision is based on sign of score
		if ( cd.raw_score )
			cd.decision = ( cd.raw_score < 0 ) ? -1 : 1;

		// decision is neutral if score doesn't exceed threshold
		if ( abs ( cd.raw_score ) < min_sentiment )
			cd.decision = 0;
	}

	return true;
}

bool SentimentClassifier::Classify (
		const string& title, const string& body,
		const string& url, CDecision& cd)
// return true if sentiment classification is successful; return false otherwise;
{
	string ncontent;
	CDecision cd_title;
	if ( normalizeContent ( title, ncontent ) )
		classifyGreedy ( TitleWeight, ncontent, cd_title );
	else return false;

	CDecision cd_body;
	if ( normalizeContent ( body, ncontent ) )
		classifyGreedy ( BodyWeight, ncontent, cd_body );
	else return false;

	CDecision cd_url;
	if ( normalizeUrl ( url, ncontent ) )
		classifyGreedy ( URLWeight, ncontent, cd_url );
	else return false;

	try {
		int min_sentiment = int ( FeatureScoreScale * NeutralCutoff );

		cd.content  =
				cd_title.content + "+ " +
				cd_body.content + "+ " +
				cd_url.content;

		cd.features.insert(cd.features.end(),
						   cd_title.features.begin(),cd_title.features.end());
		cd.features.insert(cd.features.end(),
						   cd_body.features.begin(),cd_body.features.end());
		cd.features.insert(cd.features.end(),
						   cd_url.features.begin(),cd_url.features.end());

		cd.raw_score =
				cd_title.raw_score +
				cd_body.raw_score +
				cd_url.raw_score;


		if ( cd.features.size() == 0 ) {
			cd.confidence = -1;
			error_msg = "no decision could be reached";
		} else {
			// confidence is average relevance normalized over observed features
			cd.confidence =
				(	cd_title.confidence * TitleWeight +
					cd_body.confidence * BodyWeight +
					cd_url.confidence * URLWeight 	) /
				(	TitleWeight + BodyWeight + URLWeight 	);

			// decision is based on sign of score
			if ( cd.raw_score )
				cd.decision = ( cd.raw_score < 0 ) ? -1 : 1;

			// decision is neutral if score doesn't exceed threshold
			if ( abs ( cd.raw_score ) < min_sentiment )
				cd.decision = 0;

			// decision is neutral if confidence is low
			// if ( cd.confidence < min_sentiment ) cd.decision = 0;
		}

	} catch (...) {
		cd.confidence = -1;
		error_msg = "error in SentimentClassifier::Classify";
	}

	return ( cd.confidence >= 0 );
}

bool SentimentClassifier::normalizeUrl (
		const string& content, string& ncontent)
{
	bool status = false;
	try {
		ncontent = boost::to_lower_copy ( content );

		sregex httpx = sregex::compile( "http:\\/\\/[^\\/]+\\/" );
		ncontent = regex_replace ( ncontent, httpx, "" );

		sregex punctx = sregex::compile( "[^\\w\\d]+" );
		ncontent = regex_replace ( ncontent, punctx, " " );

		status = true;
	} catch (...) {
		error_msg = "error in SentimentClassifier::normalizeUrl";
	}

	return status;
}

bool SentimentClassifier::normalizeContent (
		const string& content, string& ncontent)
// normalize punctuation and case of the content
{
	bool status = false;
	try {
		ncontent = boost::to_lower_copy ( content );

		// hash_tag hiding for tweets
        sregex hashx = sregex::compile( "^#\\S+| #\\S+" );
        ncontent = regex_replace ( ncontent, hashx, " " );

        // at_tag hiding for tweets
        sregex atx = sregex::compile( "^@\\S+| @\\S+" );
        ncontent = regex_replace ( ncontent, atx, " " );

        // hiding http tagged urls
		sregex urlx = sregex::compile( "http:[\\/\\w\\d\\.\\=\\&\\?]+" );
		ncontent = regex_replace ( ncontent, urlx, "" );

		sregex symbolx = sregex::compile( "[^a-z0-9\\']" );
		ncontent = regex_replace ( ncontent, symbolx, " " );

		sregex wsx = sregex::compile( "\\s+" );
		ncontent = regex_replace ( ncontent, wsx, " " );

		sregex trimx = sregex::compile( "^\\s+|\\s+$" );
		ncontent = regex_replace ( ncontent, trimx, "" );

		status = true;
	} catch (...) {
		error_msg = "error in SentimentClassifier::normalizeContent";
	}

	return status;
}

bool SentimentClassifier::Inited () const
// return true if the sentiment classifier is initialized properly.
{
	return isInited;
}

bool SentimentClassifier::readFeatures ( const string& features_file )
{
	bool isSuccess = false;

	string phrase, entry;
	ifstream fs ( features_file.c_str() );

	if ( fs.good() ) {
		isSuccess = true;
		while ( getline ( fs, phrase, '\t' ) ) {
			getline ( fs, entry, '\n' );
			isSuccess = parseFeature ( phrase, entry );
			if ( !isSuccess ) break;
		}
	} else {
		error_msg = "Failed to open features file.";
	}

	return isSuccess;
}

bool SentimentClassifier::parseFeature ( string& phrase, string& entry )
{
	bool isSuccess = false;

	try { 	   // phrase is added to FeatureTable

		int score_data;
		stringstream iss (entry);
		iss >> score_data;

		features[phrase].score = score_data;
		features[phrase].relevance = abs ( score_data );

		isSuccess = true;

	} catch (...) {
		error_msg = "error in SentimentClassifier::parseFeature";
	}

	return isSuccess;
}

bool SentimentClassifier::readStopwords ( const string& stopwords_file )
{
	bool isSuccess = false;

	string word;
	ifstream fs ( stopwords_file.c_str() );

	if ( fs.good() ) {
		while ( fs ) {
			fs >> word;
			stopwords.insert ( make_pair ( word, 1 ) );
		}
		isSuccess = true;
	} else {
		error_msg = "Failed to open stopwords file.";
	}

	return isSuccess;
}

void SentimentClassifier::setUseQuestionMarks ( bool qm )
{
	UseQuestionMarks = qm;
}

bool SentimentClassifier::getUseQuestionMarks () const
{
	return UseQuestionMarks;
}

void SentimentClassifier::setRelevanceCutoff ( float rc )
{
	RelevanceCutoff = rc;
}

float SentimentClassifier::getRelevanceCutoff () const
{
	return RelevanceCutoff;
}

void SentimentClassifier::setNeutralCutoff ( float nc )
{
	NeutralCutoff = nc;
}

float SentimentClassifier::getNeutralCutoff () const
{
	return NeutralCutoff;
}

void SentimentClassifier::setMaxFeatureSize ( unsigned int mfs )
{
	MaxFeatureSize = mfs;
}

unsigned int SentimentClassifier::getMaxFeatureSize () const
{
	return MaxFeatureSize;
}

void SentimentClassifier::setDebugLevel ( unsigned int dl )
{
	DebugLevel = dl;
}

unsigned int SentimentClassifier::getDebugLevel () const
{
	return DebugLevel;
}

string SentimentClassifier::getErrorMsg () const
{
	return error_msg;
}

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

bool SentimentClassifier::classifyGreedy ( int weight,
		string& content, CDecision& cd )
{

	try {
		int cutoff = int ( FeatureScoreScale * RelevanceCutoff );

		cd.content += content + "  ";

		int negative_score = 0;
		int neutral_score  = 0;
		int positive_score = 0;

		vector<string> tokens;
		boost::split ( tokens, content, boost::is_any_of ( " " ) );

		if ( DebugLevel > 1 )
			cout << "Content: " << cd.content << endl;

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
					if ( DebugLevel > 1 )
						cout << "Feature: " << ss.str();

					if ( features.find( ss.str() ) != features.end() ) {
						if ( DebugLevel > 1 )
							cout << " *Found, rc = " <<
								features[ ss.str() ].relevance;
						if ( features[ ss.str() ].relevance > cutoff ) {
							if ( DebugLevel > 1 )
								cout << " meets cutoff (" <<
									cutoff << ")" << endl;
							test_feature = ss.str();
							break;
						}
					}
					if ( DebugLevel > 1 )
						cout << endl;
				}
			}

			if ( test_feature != "" ) {
				FeatureScores* fs = &features[ test_feature ];

				cd.negative += weight * fs->negative;
				cd.neutral  += weight * fs->neutral;
				cd.positive += weight * fs->positive;
				cd.features.push_back( test_feature );

				if ( DebugLevel > 0 ) cout << test_feature <<
						" (neg=" << weight * fs->negative <<
						" neu=" << weight * fs->neutral <<
						" pos=" << weight * fs->positive << ")" << endl;

				if ( fs->neutral > fs->positive && fs->neutral > fs->negative)
					{ neutral_score++; }
				else if ( fs->positive > fs->negative )
					{ positive_score++; }
				else if ( fs->negative > fs->positive )
					{ negative_score++; }

				i += s-1;
			}
		}

		if ( cd.neutral > cd.negative && cd.neutral > cd.positive )
			{ cd.decision = 0; cd.score = neutral_score; }
		else if ( cd.positive > cd.negative )
			{ cd.decision = 1; cd.score = positive_score; }
		else if ( cd.negative > cd.positive )
			{ cd.decision = -1; cd.score = negative_score; }
		else // no decision could be reached
			{ cd.decision = 0; cd.score = -1;
			  error_msg = "no decision could be reached"; }

		if ( cd.score > 255 ) cd.score = 255;
	} catch (...) {
		cd.score = -1;
		error_msg = "error in SentimentClassifier::classifyGreedy";
	}

	return ( cd.score >= 0 );
}

CDecision::CDecision ()
	: decision(0), score(-1), negative(0), neutral(0), positive(0),
	  content(), features()
{}

FeatureScores::FeatureScores ()
	: negative(0), neutral(0), positive(0), relevance(0)
{}

SentimentClassifier::SentimentClassifier (
		const string& feature_file, const string& stopword_file)
	: RelevanceCutoff (1.0f), MaxFeatureSize (3), DebugLevel (0),
	  error_msg (), TitleWeight (3), BodyWeight (1), URLWeight (1),
	  isInited (false), features (), stopwords ()
{
	isInited =
			readFeatures (feature_file);
//			&& readStopwords (stopword_file);
}

bool SentimentClassifier::Classify (
		const string& content, CDecision& cd)
// return true if sentiment classification is successful; return false otherwise;
{
	string ncontent;
	if ( normalizeContent ( content, ncontent ) )
		return classifyGreedy ( 1, ncontent, cd );
	else return false;
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
		cd.content  = cd_title.content + cd_body.content + cd_url.content;
		cd.features.insert(cd.features.end(),
						   cd_title.features.begin(),cd_title.features.end());
		cd.features.insert(cd.features.end(),
						   cd_body.features.begin(),cd_body.features.end());
		cd.features.insert(cd.features.end(),
						   cd_url.features.begin(),cd_url.features.end());
		cd.negative = cd_title.negative + cd_body.negative + cd_url.negative;
		cd.neutral  = cd_title.neutral  + cd_body.neutral  + cd_url.neutral;
		cd.positive = cd_title.positive + cd_body.positive + cd_url.positive;

		if ( cd.neutral > cd.negative && cd.neutral > cd.positive )
			{ cd.decision = 0; cd.score = 1; }
		else if ( cd.positive > cd.negative )
			{ cd.decision = 1; cd.score = 1; }
		else if ( cd.negative > cd.positive )
			{ cd.decision = -1; cd.score = 1; }
		else // no decision could be reached
			{ cd.decision = 0; cd.score = -1;
			  error_msg = "no decision could be reached"; }

		if ( cd.score > 255 ) cd.score = 255;
	} catch (...) {
		cd.score = -1;
		error_msg = "error in SentimentClassifier::Classify";
	}

	return ( cd.score >= 0 );
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

		boost::replace_all ( ncontent, "[...]" , " " );
		boost::replace_all ( ncontent, "(...)" , " " );
		boost::replace_all ( ncontent, "^" , "" );
		boost::replace_all ( ncontent, "," , " " );
		boost::replace_all ( ncontent, "(" , " " );
		boost::replace_all ( ncontent, ")" , " " );
		boost::replace_all ( ncontent, "-" , " " );
		boost::replace_all ( ncontent, "\"" , " " );
		boost::replace_all ( ncontent, ": " , " " );

		sregex quotx1 = sregex::compile( "^'| '" );
		ncontent = regex_replace ( ncontent, quotx1, " " );

		sregex quotx2 = sregex::compile( "'$|' " );
		ncontent = regex_replace ( ncontent, quotx2, " " );

		sregex hashx = sregex::compile( "^#\\S+| #\\S+" );
		ncontent = regex_replace ( ncontent, hashx, " " );

		sregex atx = sregex::compile( "^@\\S+| @\\S+" );
		ncontent = regex_replace ( ncontent, atx, " " );

		sregex urlx = sregex::compile( "(http:[\\/\\w\\d\\.\\=\\&\\?]+)" );

		sregex_iterator cur( ncontent.begin(), ncontent.end(), urlx );
		sregex_iterator end;

		for( ; cur != end; ++cur ) {
			boost::replace_all (
					ncontent, cur->str(), boost::to_upper_copy (cur->str()) );
		}

		sregex perx1 = sregex::compile( "\\s*\\.\\.+\\s*" );
		ncontent = regex_replace ( ncontent, perx1, " " );

		sregex perx2 = sregex::compile( "([ a-z])\\.([ a-z])" );
		ncontent = regex_replace ( ncontent, perx2, "$1 $2" );

		sregex slashx = sregex::compile( "([ a-z])\\/([ a-z])" );
		ncontent = regex_replace ( ncontent, slashx, "$1 $2" );

		sregex uscorex = sregex::compile( "([ a-z])_([ a-z])" );
		ncontent = regex_replace ( ncontent, uscorex , "$1 $2" );

		sregex endx = sregex::compile( "\\W*$" );
		ncontent = regex_replace ( ncontent, endx, "" );

		sregex spacex = sregex::compile( "\\s+" );
		ncontent = regex_replace ( ncontent, spacex, " " );

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

	if 			( phrase == "CLASS:" )  { isSuccess = true; }
	else if 	( phrase == "TOTAL:" )  { isSuccess = true; }
	else if 	( phrase == "WTOTAL:" ) { isSuccess = true; }
	else { 	   // phrase is added to FeatureTable
		float raw_negative;
		float raw_neutral;
		float raw_positive;

		stringstream iss (entry);
		iss >> raw_negative;
		iss >> raw_neutral;
		iss >> raw_positive;

		int negative;
		int neutral;
		int positive;

		isSuccess =
				scaleRawValue (raw_negative, negative) &&
				scaleRawValue (raw_neutral,  neutral) &&
				scaleRawValue (raw_positive, positive);

		if ( isSuccess ) {
			features[phrase].negative = negative;
			features[phrase].neutral  = neutral;
			features[phrase].positive = positive;

			features[phrase].relevance = abs ( positive - negative );
		} else {
			error_msg = "Failed to parse feature line.";
		}
	}

	return isSuccess;
}

bool SentimentClassifier::scaleRawValue (
		float raw_value, int& scaled_value )
{
	if ( raw_value > 0.f ) {
		scaled_value = (int) floor( FeatureScoreScale * log( raw_value ) );
		return true;
	} else {
		return false;
	}
}

float SentimentClassifier::getRawValue ( int scaled )
{
	return exp ( (float) scaled / FeatureScoreScale );
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

void SentimentClassifier::setRelevanceCutoff ( float rc )
{
	RelevanceCutoff = rc;
}

float SentimentClassifier::getRelevanceCutoff () const
{
	return RelevanceCutoff;
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

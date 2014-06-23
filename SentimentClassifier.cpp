/*
 * SentimentClassifier.cpp
 *
 *  Created on: Dec 25, 2009
 *      Author: Christopher L. Tang
 */

#include "SentimentClassifier.h"

#include <assert.h>

#include <math.h>
#include <fstream>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <boost/xpressive/xpressive.hpp>

using namespace boost::xpressive;

int calc_confidence ( int raw_score )
{
	int confidence = 50 + raw_score / 70;
	if ( confidence > 100 ) confidence = 100;
	if ( confidence <   1 ) confidence =   1;
	return confidence;
}

int calc_decision ( int raw_score, float nc )
{
	int decision = 0;
	int min_sentiment = int ( SentimentClassifier::FeatureScoreScale * nc );

	if ( raw_score <= -min_sentiment ) decision = -1;
	if ( raw_score >=  min_sentiment ) decision =  1;

	return decision;
}

void collapse_ws ( string& ncontent )
{
	sregex wsx = sregex::compile( "\\s+" );
	ncontent = regex_replace ( ncontent, wsx, " " );
}

int countFeatures ( const string& content, const string& feature )
{
	int count = 0;
	typedef boost::find_iterator<string::const_iterator> string_find_iterator;
    for ( string_find_iterator it=
            boost::make_find_iterator ( content,
            		boost::first_finder ( feature, boost::is_iequal() ) );
    		it!=string_find_iterator (); ++it ) ++count;
    return count;
}

void hide_urls ( string& ncontent )
{
	sregex urlx = sregex::compile ( "http:[_\\/\\w\\d\\.\\=\\&\\?\\-]+" );
	ncontent = regex_replace ( ncontent, urlx, "'http'" );
}

void trim_ws ( string& ncontent )
{
	boost::trim ( ncontent );
}

void xlate_html_entities ( string& ncontent )
{
	boost::replace_all ( ncontent, "&quot;"  , "\"" );
	boost::replace_all ( ncontent, "&amp;"   , "&" );
	boost::replace_all ( ncontent, "&lt;"    , ">" );
	boost::replace_all ( ncontent, "&gt;"    , "<" );
	boost::replace_all ( ncontent, "&nbsp;"  , " " );
	boost::replace_all ( ncontent, "&#34;"   , "\"" );
	boost::replace_all ( ncontent, "&#034;"  , "\"" );
	boost::replace_all ( ncontent, "&#35;"   , "#" );
	boost::replace_all ( ncontent, "&#035;"  , "#" );
	boost::replace_all ( ncontent, "&#39;"   , "'" );
	boost::replace_all ( ncontent, "&#039;"  , "'" );
	boost::replace_all ( ncontent, "&#96;"   , "'" );
	boost::replace_all ( ncontent, "&#8211;" , "--" );
	boost::replace_all ( ncontent, "&#8212;" , "--" );
	boost::replace_all ( ncontent, "&#8220;" , "..." );
	boost::replace_all ( ncontent, "&#8230;" , "..." );
	boost::replace_all ( ncontent, "&heart;" , "'heart'" );

	boost::replace_all ( ncontent, "\\u00b4" , "'" );
	boost::replace_all ( ncontent, "\\u2019" , "'" );
}

void SentimentClassifier::classifyExclamationPoints (
		int weight, const string& ucontent, CDecision& cd )
{
	string feature ( ucontent );

	sregex non_epx = sregex::compile( "[^!]" );
	feature = regex_replace ( feature, non_epx, "" );

	float epc = float ( feature.size() );
	epc = epc > 20 ? 20 : epc;

	float raw_score = 0.f;

	if ( feature.size() > 0 )
		raw_score = SentimentClassifier::FeatureScoreScale *
										 ( 0.002f * epc * epc
										 - 0.083f * epc + 0.87f );

	cd.raw_score = weight * int ( raw_score );
	if ( feature.size() > 0 ) {
		if ( DebugLevel > 0 ) {
			stringstream ss;
			ss << "!: '" << feature << "' = " << int ( raw_score );
			cd.features.push_back( ss.str() );
		} else {
			cd.features.push_back( feature );
		}
	}
}

void SentimentClassifier::classifyQuestionMarks (
		int weight, const string& ucontent, CDecision& cd )
{
	string feature ( ucontent );

	hide_urls ( feature );

	float qm_ratio = float ( feature.size() );

	sregex non_qmx = sregex::compile( "[^\\?]" );
	feature = regex_replace ( feature, non_qmx, "" );

	qm_ratio = float ( feature.size() ) / qm_ratio;
	float raw_score = 0.f;
	if ( qm_ratio > 0.001f )
		raw_score = FeatureScoreScale * ( -156.f * qm_ratio - 0.3f );

	//float qm_count = float ( feature.size() );
	//float raw_score = FeatureScoreScale * -0.37f * sqrt ( qm_count );

	cd.raw_score = weight * int ( raw_score );
	if ( qm_ratio > 0.f ) {
		if ( DebugLevel > 0 ) {
			stringstream ss;
			ss << "?: '" << feature << "' = " << int ( raw_score );
			cd.features.push_back( ss.str() );
		} else {
			cd.features.push_back( feature );
		}
	}
}

void SentimentClassifier::classifyTweetQuestionMarks (
		int weight, const string& ucontent, CDecision& cd )
{
	string feature ( ucontent );

	hide_urls ( feature );

	sregex non_qmx = sregex::compile( "[^\\?]" );
	feature = regex_replace ( feature, non_qmx, "" );

	float raw_score = 0.f;
	if ( feature.size() > 0 )
		raw_score = FeatureScoreScale *
			( -0.5533f * log ( float ( feature.size() ) ) - 0.3533f );

	cd.raw_score = weight * int ( raw_score );
	if ( raw_score < 0.f ) {
		if ( DebugLevel > 0 ) {
			stringstream ss;
			ss << "?: '" << feature << "' = " << int ( raw_score );
			cd.features.push_back( ss.str() );
		} else {
			cd.features.push_back( feature );
		}
	}
}

bool SentimentClassifier::classifySentences ( int weight,
		const string& ucontent, CDecision& cd,
		int ct, float rc, float nc )
{
	string content ( ucontent );

	hide_urls ( content );
	xlate_html_entities ( content );

	vector<string> sentences;
	boost::split ( sentences, content, boost::is_any_of ( ";?!" ) );

	for ( vector<string>::iterator sentence = sentences.begin();
			sentence != sentences.end(); sentence++ ) {
		CDecision cd_s;
		string nsentence;

		if ( normalizeContent ( *sentence, nsentence ) )
			classifyGreedy ( weight, nsentence, cd_s, ct, rc, nc );
		else return false;

		if ( DebugLevel > 0 ) cd.content += ";; " + nsentence;
		cd.raw_score += cd_s.raw_score;
		cd.features.insert(cd.features.end(),
						   cd_s.features.begin(),cd_s.features.end());
	}

	if ( cd.features.size() == 0 ) {
		cd.confidence = 0;
		error_msg = "no decision could be reached";
		return false;
	} else {
		cd.decision   = calc_decision   ( cd.raw_score, nc );
		cd.confidence = calc_confidence ( cd.raw_score );
		return true;
	}
}

bool SentimentClassifier::classifyGreedy ( int weight,
		string& content, CDecision& cd, int ct, float rc, float nc )
{
	// Split content to obtain tokens
	vector<string> tokens;
	boost::split ( tokens, content, boost::is_any_of ( " " ) );

	// Limit the size of the content
	if ( tokens.size() > 500 )
		tokens.erase ( tokens.begin(), tokens.end() - 500 );

	// Set up table to count repeated features
	FeaturesCount fc;

	// Create scaled relevance cutoff
	int cutoff = int ( FeatureScoreScale * rc );

	// Algorithm for extracting features from content
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

				if      ( ct == Regular ) ss << ":r" ;
				else if ( ct == Twitter ) ss << ":t" ;
				else {
					error_msg = "invalid classification type";
					return false ;
				}

				// if ( DebugLevel > 2 ) cout << "Feature? " << ss.str();

				if ( features.find ( ss.str() ) != features.end() ) {
					// if ( DebugLevel > 2 ) cout << "; YES rc = " << features[ ss.str() ].relevance;
					if ( features[ ss.str() ].relevance > cutoff ) {
						// if ( DebugLevel > 2 ) cout << "; PASSES cutoff (" << cutoff << ")" << endl;
						test_feature = ss.str();
						break;
					}
				} // else { if ( DebugLevel > 2 ) cout << "; NO"; }

				// if ( DebugLevel > 2 ) cout << endl;
			}
		}

		if ( test_feature != "" ) {

			if ( fc.find ( test_feature ) == fc.end() )
				fc[ test_feature ] = 0;
			fc[ test_feature ] ++;

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
		cd.confidence = 0;
		error_msg = "no decision could be reached";
		return false;
	} else {
		cd.decision   = calc_decision   ( cd.raw_score, nc );
		cd.confidence = calc_confidence ( cd.raw_score );
		return true;
	}
}

CDecision::CDecision ()
	: decision(0), raw_score(0), confidence(0), content(), features()
{}

FeatureScores::FeatureScores ()
	: score(0), relevance(0)
{}

SentimentClassifier::SentimentClassifier (
		const string& feature_file, const string& stopword_file)
	: UseQuestionMarks (true), UseExclamationPoints (true), UseEmoticons (true),
	  RelevanceCutoff (-1.f), NeutralCutoff (-1.f), MaxFeatureSize (7),
	  DebugLevel (0), error_msg (), TitleWeight (3), BodyWeight (1),
	  URLWeight (1), isInited (false), features (), stopwords (),
	  pre_normalized (false)
{
	isInited =
			readFeatures (feature_file);
//			&& readStopwords (stopword_file);
}

bool SentimentClassifier::Classify (
		const string& content, CDecision& cd, int contentType )
// return true if sentiment classification is successful; return false otherwise;
{
	float nc = NeutralCutoff;
	float rc = RelevanceCutoff;

	if ( contentType == Twitter && ( rc < 0 || nc < 0 ) ) {
		rc = 1.f; nc = 5.f;
	} else if ( contentType == Regular && ( rc < 0 || nc < 0 ) ) {
		rc = 1.f; nc = 5.f;
	}

	cd.content = "\"" + content + "\"";

	if ( contentType == Regular ) {
		classifySentences ( 1, content, cd, contentType, rc, nc );
	} else if ( contentType == Twitter ) {
		string ncontent;
		if ( ! normalizeTweet ( content, ncontent ) ) return false;
		if ( DebugLevel > 0 ) cd.content += ";; " + ncontent;
		classifyGreedy ( 1, ncontent, cd, contentType, rc, nc );
	}

	if ( UseEmoticons ) {
		int count_pos;

		count_pos  = countFeatures ( content, ":)" ) ;
		count_pos += countFeatures ( content, ": )" ) ;
		count_pos += countFeatures ( content, ":-)" ) ;
		count_pos += countFeatures ( content, ":D" ) ;
		count_pos += countFeatures ( content, "=)" ) ;
		count_pos += countFeatures ( content, "(:" ) ;
		count_pos += countFeatures ( content, ";)" );
		count_pos += countFeatures ( content, ";-)" );
		count_pos += countFeatures ( content, ";-)" );
		count_pos += countFeatures ( content, ":]" );

		count_pos += countFeatures ( content, "<3" );
		count_pos += countFeatures ( content, "&lt;3" );

		int count_neg;
		count_neg  = countFeatures ( content, ":(" ) ;
		count_neg += countFeatures ( content, ": (" ) ;
		count_neg += countFeatures ( content, ":-(" ) ;
		count_neg += countFeatures ( content, "):" ) ;
		count_neg += countFeatures ( content, ":[" ) ;

		int count = count_pos - count_neg;

		if ( count != 0 ) {
			cd.raw_score += count * 1000 ;

			stringstream ss;
			ss << ":) *" << count_pos << "; ";
			ss << ":( *" << count_neg ;
			cd.features.push_back ( ss.str() );
		}
	}

	if ( UseExclamationPoints ) {
		CDecision cd_ep;
		classifyExclamationPoints ( 1, content, cd_ep );

		if ( cd_ep.features.size () == 1 )
			cd.features.push_back( cd_ep.features[0] );
		cd.raw_score += cd_ep.raw_score;
	}

	if ( UseQuestionMarks ) {
		CDecision cd_qm;
		if ( contentType == Regular )
			classifyQuestionMarks ( 1, content, cd_qm );
		else if ( contentType == Twitter )
			classifyTweetQuestionMarks ( 1, content, cd_qm );

		if ( cd_qm.features.size () == 1 )
			cd.features.push_back ( cd_qm.features[0] );
		cd.raw_score += cd_qm.raw_score;
	}


	if ( cd.features.size() == 0 ) {
		cd.confidence = 0;
		error_msg = "no decision could be reached";
		return false;
	} else {
		cd.decision   = calc_decision   ( cd.raw_score, nc );
		cd.confidence = calc_confidence ( cd.raw_score );
		return true;
	}
}

bool SentimentClassifier::Classify (
		const string& title, const string& body,
		const string& url, CDecision& cd, int contentType )
// return true if sentiment classification is successful; return false otherwise;
{
	float nc = NeutralCutoff;
	float rc = RelevanceCutoff;
	if ( contentType == Twitter && ( rc < 0 || nc < 0 ) )
		{ rc = 1.f; nc = 5.f; }
	else if ( contentType == Regular && ( rc < 0 || nc < 0 ) )
		{ rc = 1.f; nc = 5.f; }

	cd.content  = "\"" + title + "\"+" +
			"\"" + body + "\"+" +
			"\"" + url + "\"" ;

	string ncontent;
	CDecision cd_title;
	if ( normalizeContent ( title, ncontent ) ) {
		cd.content += ";; " + ncontent;
		classifyGreedy ( TitleWeight, ncontent, cd_title, contentType, rc, nc );
	}
	else return false;

	CDecision cd_body;
	if ( normalizeContent ( body, ncontent ) ) {
		cd.content += ";; " + ncontent;
		classifyGreedy ( BodyWeight, ncontent, cd_body, contentType, rc, nc );
	}
	else return false;

	CDecision cd_url;
	if ( normalizeUrl ( url, ncontent ) ) {
		cd.content += ";; " + ncontent;
		classifyGreedy ( URLWeight, ncontent, cd_url, contentType, rc, nc );
	}
	else return false;

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
		cd.confidence = 0;
		error_msg = "no decision could be reached";
		return false;
	} else {
		cd.decision   = calc_decision   ( cd.raw_score, nc );
		cd.confidence = calc_confidence ( cd.raw_score );
		return true;
	}
}

bool SentimentClassifier::normalizeUrl (
		const string& content, string& ncontent)
{
	bool status = false;
	try {
		ncontent = boost::to_lower_copy ( content );

		sregex httpx = sregex::compile( "https?://[^/]+/" );
		ncontent = regex_replace ( ncontent, httpx, "" );

		sregex htmlx = sregex::compile( "\\.html$" );
		ncontent = regex_replace ( ncontent, htmlx, " " );

		sregex punctx = sregex::compile( "[^a-zA-Z0-9]+" );
		ncontent = regex_replace ( ncontent, punctx, " " );

		status = true;
	} catch (...) {
		error_msg = "error in SentimentClassifier::normalizeUrl";
	}

	return status;
}

bool SentimentClassifier::normalizeTweet (
		const string& content, string& ncontent )
// normalize tweets
{
	ncontent = content;
	if ( pre_normalized ) return true;

	bool status = false;

	try {
		sregex rx01 = sregex::compile ( "(?<!\\\\)\\\\n" );
		sregex rx02 = sregex::compile ( "(?<!\\\\)\\\\r" );
		sregex rx03 = sregex::compile ( "(?<!\\\\)\\\\t" );

		ncontent = regex_replace ( ncontent, rx01, " " );
		ncontent = regex_replace ( ncontent, rx02, " " );
		ncontent = regex_replace ( ncontent, rx03, " " );

		hide_urls ( ncontent );
		xlate_html_entities ( ncontent );

		/*
		sregex rx04 = sregex::compile ( "(?<![AQ]):\\)+" );
		sregex rx05 = sregex::compile ( "(?<![AQ]): \\)+" );
		sregex rx06 = sregex::compile ( "(?<![AQ]):-\\)+" );
		//sregex rx17 = sregex::compile ( "(?<!Q):\\]+" );
		sregex rx07 = sregex::compile ( "(?<![AQ]):D+" );
		sregex rx08 = sregex::compile ( "=\\)+" );
		sregex rx09 = sregex::compile ( "\\(+:" );

		ncontent = regex_replace ( ncontent, rx04, "'smile'" );
		ncontent = regex_replace ( ncontent, rx05, "'smile'" );
		ncontent = regex_replace ( ncontent, rx06, "'smile'" );
		//ncontent = regex_replace ( ncontent, rx17, "'smile'" );
		ncontent = regex_replace ( ncontent, rx07, "'smile'" );
		ncontent = regex_replace ( ncontent, rx08, "'smile'" );
		ncontent = regex_replace ( ncontent, rx09, "'smile'" );

		sregex rx10 = sregex::compile ( ";\\)+" );
		sregex rx11 = sregex::compile ( ";-\\)+" );
		ncontent = regex_replace ( ncontent, rx10, "'wink'" );
		ncontent = regex_replace ( ncontent, rx11, "'wink'" );

		sregex rx12 = sregex::compile ( "<3+" );
		ncontent = regex_replace ( ncontent, rx12, "'heart'" );

		sregex rx13 = sregex::compile ( "(?<![AQ]):\\(+" );
		sregex rx14 = sregex::compile ( "(?<![AQ]): \\(+" );
		sregex rx15 = sregex::compile ( "(?<![AQ]):-\\(+" );
		//sregex rx18 = sregex::compile ( "(?<!Q):\\[+" );
		sregex rx16 = sregex::compile ( "\\)+:" );

		ncontent = regex_replace ( ncontent, rx13, "'unsmile'" );
		ncontent = regex_replace ( ncontent, rx14, "'unsmile'" );
		ncontent = regex_replace ( ncontent, rx15, "'unsmile'" );
		ncontent = regex_replace ( ncontent, rx16, "'unsmile'" );
		//ncontent = regex_replace ( ncontent, rx18, "'unsmile'" );
		*/

		// ** everything below should be case insensitive **

		boost::to_lower ( ncontent );

		sregex atx = sregex::compile ( "@[_a-z0-9]+" );
		ncontent = regex_replace ( ncontent, atx, "'atdel'" );

		sregex hashx = sregex::compile ( "#[_a-z0-9]+" );
		ncontent = regex_replace ( ncontent, hashx, "'hashdel'" );

		sregex ucx = sregex::compile ( "\\\\u[0-9a-f]{4}" );
		ncontent = regex_replace ( ncontent, ucx, "'u'" );

		sregex numx = sregex::compile ( "[,0-9\\.]*[0-9]" );
		ncontent = regex_replace ( ncontent, numx, "'n'" );

		// collapse multi-letter tweet content ( e.g. loooove! -> loove )
		sregex tweetx = sregex::compile ( "(\\w)\\1+" );
		ncontent = regex_replace ( ncontent, tweetx, "$1$1" );

		sregex rtx = sregex::compile ( "\\brt " );
		ncontent = regex_replace ( ncontent, rtx, "" );

		sregex symbolx = sregex::compile( "[^a-z0-9\\']" );
		ncontent = regex_replace ( ncontent, symbolx, " " );

		collapse_ws ( ncontent );
		trim_ws ( ncontent );

		status = true;

	} catch (...) {
		error_msg = "error in SentimentClassifier::normalizeTweet";
	}

	return status;
}

bool SentimentClassifier::normalizeContent (
		const string& content, string& ncontent )
// normalize punctuation and case of the content
{
	if ( pre_normalized ) {
		ncontent = content;
		return true;
	}

	bool status = false;
	try {
		ncontent = boost::to_lower_copy ( content );

		// hash_tag hiding for tweets
        sregex hashx = sregex::compile( "^#\\S+| #\\S+" );
        ncontent = regex_replace ( ncontent, hashx, " " );

        // at_tag hiding for tweets
        sregex atx = sregex::compile( "^@\\S+| @\\S+" );
        ncontent = regex_replace ( ncontent, atx, " " );

        hide_urls ( ncontent );

		// collapse multi-letter tweet content ( e.g. loooove! -> loove )
		sregex tweetx = sregex::compile ( "(\\w)\\1+" );
		ncontent = regex_replace ( ncontent, tweetx, "$1$1" );

		sregex rtx = sregex::compile ( "\\brt " );
		ncontent = regex_replace ( ncontent, rtx, "" );

		sregex symbolx = sregex::compile( "[^a-z0-9\\']" );
		ncontent = regex_replace ( ncontent, symbolx, " " );

		collapse_ws ( ncontent );
		trim_ws ( ncontent );


		status = true;
	} catch (...) {
		error_msg = "error in SentimentClassifier::normalizeContent";
	}

	return status;
}

bool SentimentClassifier::parseFeature ( string& phrase, string& entry )
{
	bool isSuccess = false;

	try { 	   // phrase is added to FeatureTable

		int score_data;
		string feature_type = "";
		stringstream iss (entry);
		iss >> score_data;
		if ( ! iss.eof() ) iss >> feature_type;

		string feature_name = phrase;
		if      ( feature_type == "re" ) feature_name += ":r";
		else if ( feature_type == "tw" ) feature_name += ":t";
		else      feature_name += ":r";

		features[feature_name].score = score_data;
		features[feature_name].relevance = abs ( score_data );

		isSuccess = true;

	} catch (...) {
		error_msg = "error in SentimentClassifier::parseFeature";
	}

	return isSuccess;
}

bool SentimentClassifier::readFeatures ( const string& features_file )
{
	string phrase, entry;
	ifstream fs ( features_file.c_str() );

	if ( fs.good() ) {
		while ( getline ( fs, phrase, '\t' ) ) {
			getline ( fs, entry, '\n' );
			if ( ! parseFeature ( phrase, entry ) ) return false;
		}
		return true;
	} else {
		error_msg = "Failed to open features file.";
		return false;
	}
}

bool SentimentClassifier::readStopwords ( const string& stopwords_file )
{
	string word;
	ifstream fs ( stopwords_file.c_str() );

	if ( fs.good() ) {
		while ( fs ) {
			fs >> word;
			stopwords.insert ( make_pair ( word, 1 ) );
		}
		return true;
	} else {
		error_msg = "Failed to open stopwords file.";
		return false;
	}
}

bool SentimentClassifier::Inited () const
// return true if the sentiment classifier is initialized properly.
{
	return isInited;
}

void SentimentClassifier::setUseQuestionMarks ( bool qm )
{
	UseQuestionMarks = qm;
}

bool SentimentClassifier::getUseQuestionMarks () const
{
	return UseQuestionMarks;
}

void SentimentClassifier::setUseExclamationPoints ( bool ep )
{
	UseExclamationPoints = ep;
}

bool SentimentClassifier::getUseExclamationPoints () const
{
	return UseExclamationPoints;
}

void SentimentClassifier::setPreNormalized ( bool pn )
{
	pre_normalized = pn;
}

bool SentimentClassifier::getPreNormalized () const
{
	return pre_normalized;
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

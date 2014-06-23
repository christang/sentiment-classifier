/*
 * SentimentClassifierTester.cpp
 *
 *  Created on: Dec 25, 2009
 *      Author: Christopher L. Tang
 */

#include <string>
#include <fstream>
#include <iostream>
#include <vector>

// See: http://tclap.sourceforge.net/
#include <tclap/CmdLine.h>

#include "SentimentClassifier.h"

using namespace std;

void print ( CDecision& cd )
// write decision using tab-separated format
// column 1: normalized content
// column 2: feature set used to make decision
// column 3: decision from [-1, 0, +1] ( score in parens )
{
	cout << "\"" << cd.content << "\"\t";

	cout << "( ";
	if ( cd.features.size()>0 )
		for ( unsigned int i = 0; i < cd.features.size(); ++i )
			cout << cd.features[i] << "; ";
	cout << ")\t";

	if ( cd.confidence < 0 ) cout << "=";
	else {
		if ( cd.decision == 0 ) cout << "0 ";
		else if ( cd.decision == -1 ) cout << "-1 ";
		else if ( cd.decision == +1 ) cout << "+1 ";
		cout << "( raw=" << cd.raw_score <<
				"; norm=" << cd.confidence << " )";
	}

	cout << endl;
}

void split ( vector<string>& strs, string& input, char delim )
// quick implementation of boost split
{
	string elem;
	stringstream str (input);
	while ( getline ( str, elem, delim ) )
		strs.push_back ( elem );
}

bool getContent( string& inputLine, string& content )
// gets content from input line
{
	vector<string> strs;
	split ( strs, inputLine, char(9) );
	if (strs.size()==10) {
		content = strs[9];
		return true;
	} else if ( strs.size()==1 ) {
		content = strs[0];
		return true;
	} else {
		return false;
	}
}

bool getContent( string& inputLine, string& title, string& body,
		string& url )
// gets title, body and url from input line
{
	vector<string> strs;
	split(strs, inputLine, char(9) );
	if (strs.size()==10) {
		title = strs[8];
		body = strs[9];
		url = strs[0];
		return true;
	} else if ( strs.size()==3 ){
		title = strs[0];
		body = strs[1];
		url = strs[2];
		return true;
	} else {
		return false;
	}
}

int main(int argc, char **argv)
{
	const char* DescriptionMessage =
		"Provides access to methods of SentimentClassifier class from the "
		"command line";

	// filenames for files containing features & stopwords
	string features_fn;
	string stopwords_fn;  // this is currently ignored!

	// various defaults, can be changed
	unsigned int debug_level = 1;
	float relevance_cutoff = 1.0f;
	float neutral_cutoff = 1.0f;
	bool title_body_url = false;
	bool question_marks = false;
	istream *in = &cin;

	// various defaults, fixed
	unsigned int max_feature_size = 3;

	try {

		TCLAP::CmdLine cmd(
				DescriptionMessage, ' ', "1.1.3");

		TCLAP::ValueArg<std::string> inputFilenameArg(
				"c","classify","Input file (one text per line, tab-separated)",
				false,"","string",cmd);

		TCLAP::ValueArg<std::string> featuresFilenameArg(
				"f","features","Features file to use" ,true,"","string",
				cmd);

		TCLAP::ValueArg<std::string> stopwordsFilenameArg(
				"s","stopwords","Stopwords file to use [CURRENTLY IGNORED]",
				false,"","string",cmd);

		TCLAP::ValueArg<unsigned int> debugLevelArg(
				"d","debug","Level of debug info to produce",false,debug_level,
				"unsigned int",cmd);

		TCLAP::ValueArg<unsigned int> maxFeatureSizeArg(
				"m","max_feature_size","Max number of tokens in any feature",
				false,max_feature_size,"unsigned int",cmd);

		TCLAP::ValueArg<float> relevanceCutoffArg(
				"r","relevance_cutoff","Relevance cutoff of feature set",
				false,relevance_cutoff,"float",cmd);

		TCLAP::ValueArg<float> neutralCutoffArg(
				"n","neutral_cutoff","Neutral cutoff for classifier",
				false,neutral_cutoff,"float",cmd);

		TCLAP::SwitchArg titleBodyUrlSwitch(
				"t","title_body_url","Classify by title, body, and URL",
				cmd,false);

		TCLAP::SwitchArg questionMarksSwitch(
				"q","question_marks","Use question marks to classify",
				cmd,false);

		cmd.parse( argc, argv );

		features_fn    = featuresFilenameArg.getValue();
		stopwords_fn   = stopwordsFilenameArg.getValue();
		if ( inputFilenameArg.isSet() ) {
			in = new ifstream ( inputFilenameArg.getValue().c_str() );
		}
		if ( titleBodyUrlSwitch.isSet() )
			title_body_url = titleBodyUrlSwitch.getValue();

		if ( debugLevelArg.isSet() )
			debug_level = debugLevelArg.getValue();

		if ( maxFeatureSizeArg.isSet() )
			max_feature_size = maxFeatureSizeArg.getValue();

		if ( relevanceCutoffArg.isSet() )
			relevance_cutoff = relevanceCutoffArg.getValue();

		if ( neutralCutoffArg.isSet() )
			neutral_cutoff = neutralCutoffArg.getValue();

		if ( questionMarksSwitch.isSet() )
			question_marks = questionMarksSwitch.getValue();

	} catch (TCLAP::ArgException &e) {

		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
		return 1;

	}

	// Instantiate classifier, set parameters
	SentimentClassifier classifier( features_fn, stopwords_fn );

	// If DebugLevel == 0, classifier generates no msgs to stdout/stderr
	classifier.setDebugLevel ( debug_level );

	// MaxFeatureSize is the max N-gram size in feature set
	classifier.setMaxFeatureSize ( max_feature_size );

	// RelevanceCutoff is the minimum relevance to use in feature set
	classifier.setRelevanceCutoff ( relevance_cutoff );

	// NeutralCutoff is the abs raw score less than which decision is neutral
	classifier.setNeutralCutoff ( neutral_cutoff );

	// Sets whether question marks should be used as a feature
	classifier.setUseQuestionMarks( question_marks );

	// Inited checks where files are properly loaded
	if ( classifier.Inited() ) {

		// Loop over inputs
		while ( in->good() ) {

			CDecision decision;
			string inputLine;

			getline ( *in, inputLine );

			if ( inputLine.length() == 0 )
				break;

			if ( title_body_url ) {

				// Do title-body-url classification

				string title, body, url;

				if ( getContent ( inputLine, title, body, url ) ) {
					classifier.Classify ( title, body, url, decision);
					print ( decision );
				} else
					cerr << "Error parsing title, body and url! (\"" <<
					inputLine << "\")" << endl;

			} else {

				// Do single content classification

				string content;

				if ( getContent ( inputLine, content ) ) {
					classifier.Classify ( content, decision );
					print ( decision );
				} else
					cerr << "Error parsing content! (\"" <<
					inputLine << "\")"<< endl;

			}

			if ( debug_level > 1 ) cout << endl;

		}

		return 0;

	} else {

		cerr << "Classifier failed to initialize!" << endl;
		return 1;

	}

}

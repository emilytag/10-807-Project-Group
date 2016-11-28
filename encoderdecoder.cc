#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/cfsm-builder.h"
#include "dynet/hsm-builder.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <ctime>
#include <cstdio>
#include <string>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

using namespace std;
using namespace dynet;

dynet::Dict sourceD;
dynet:: Dict targetD;
int kSOS;
int kSOS2;
int kEOS;
int kEOS2;

typedef pair<float,int> prob_in_pair;

bool comparator ( const prob_in_pair& l, const prob_in_pair& r)
   { return l.first > r.first; }

unsigned IN_LAYERS = 1;
unsigned OUT_LAYERS = 2;
unsigned INPUT_DIM = 100;
unsigned HIDDEN_DIM = 400;
unsigned VOCAB_SIZE = 0;

template <class Builder>
struct EncoderDecoder {
	LookupParameter p_c;
	Parameter p_enc2dec;
	Parameter p_decbias;
	Parameter p_enc2out;
	Parameter p_outbias;
	Builder r2lbuilder;
	Builder l2rbuilder;
	Builder outbuilder;

	explicit EncoderDecoder(Model& model): l2rbuilder(IN_LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
	r2lbuilder(IN_LAYERS, INPUT_DIM, HIDDEN_DIM, &model), outbuilder(OUT_LAYERS, INPUT_DIM, HIDDEN_DIM, &model) {
		p_c = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM});
		p_enc2dec = model.add_parameters({HIDDEN_DIM * OUT_LAYERS, 2*HIDDEN_DIM});
		p_decbias = model.add_parameters({HIDDEN_DIM * OUT_LAYERS});
		p_enc2out = model.add_parameters({VOCAB_SIZE, HIDDEN_DIM});
		p_outbias = model.add_parameters({VOCAB_SIZE});
	}
    Expression Encoder(ComputationGraph& cg, const vector<int>& tokens){
    	const unsigned slen = tokens.size();
    	l2rbuilder.new_graph(cg);
    	l2rbuilder.start_new_sequence();
	l2rbuilder.add_input(lookup(cg, p_c, kSOS));
    	vector<Expression> left;

    	for (unsigned t = 0; t < slen; ++t) {
		//cerr << "==" << sourceD.convert(tokens[t])<< "\n";
    		Expression word_lookup = lookup(cg, p_c, tokens[t]);
    		l2rbuilder.add_input(word_lookup);
    		left.push_back(l2rbuilder.back());
    	}

	r2lbuilder.new_graph(cg);
    	r2lbuilder.start_new_sequence();
	r2lbuilder.add_input(lookup(cg, p_c, kSOS));
    	deque<Expression> right;

    	for (unsigned t = 0; t < slen; ++t) {
		//cerr << "==" << sourceD.convert(tokens[slen - t - 1])<< "\n";
    		Expression word_lookup = lookup(cg, p_c, tokens[slen - t - 1]);
    		r2lbuilder.add_input(word_lookup);
    		right.push_front(r2lbuilder.back());
    	}


	Expression left_norm = sum(left) / ((float)left.size());
    	Expression right_norm = sum(right) / ((float)right.size());
    	return concatenate({left_norm, right_norm});
    }
    Expression BuildGraph(const vector<int>& source_tokens, const vector<int>& target_tokens, ComputationGraph& cg) {
    	//cerr << source_tokens.size() << "\n";

    	const unsigned target_len = target_tokens.size();
    	Expression encoding = Encoder(cg, source_tokens);
    	Expression enc2dec = parameter(cg, p_enc2dec);
    	Expression decbias = parameter(cg, p_decbias);
    	Expression outbias = parameter(cg, p_outbias);
    	Expression enc2out = parameter(cg, p_enc2out);
    	Expression c0 = affine_transform({decbias, enc2dec, encoding});
    	vector<Expression> init(OUT_LAYERS * 2);
    	//initialize decoder
    	for (unsigned i = 0; i < OUT_LAYERS; ++i) {
    	      init[i] = pickrange(c0, i * HIDDEN_DIM, i * HIDDEN_DIM + HIDDEN_DIM);
    	      init[i + OUT_LAYERS] = tanh(init[i]);
    	    }
    	//vector<float> ok = as_vector(cg.incremental_forward());
    	//for (std::vector<float>::const_iterator i = ok.begin(); i != ok.end(); ++i)
    	//    std::cerr << *i << ' ';
    	//cerr << "\n";
    	outbuilder.new_graph(cg);
    	outbuilder.start_new_sequence(init);
    	vector<Expression> errs(target_len + 1);
    	Expression h_t = outbuilder.add_input(lookup(cg, p_c, kSOS2));
    	for (unsigned t = 0; t < target_len; ++t) {
    		//Expression outback = outbuilder.back();
		//cerr << "==" << targetD.convert(target_tokens[t])<< "\n";
    		Expression affine_outback = affine_transform({outbias, enc2out, h_t});
    		errs[t] = pickneglogsoftmax(affine_outback, target_tokens[t]);
    		Expression x_t = lookup(cg, p_c, target_tokens[t]);
    		h_t = outbuilder.add_input(x_t);
    	}
    	//Expression outback_last = outbuilder.back();
    	Expression affine_outback_last = affine_transform({outbias, enc2out, h_t});
    	errs.back() = pickneglogsoftmax(affine_outback_last, kEOS2);
    	Expression final_errs = sum(errs);
    	return final_errs;
    }
};

vector<string> Translate(vector<vector<int>>&  test_sents, EncoderDecoder<LSTMBuilder>& tr){
	vector<string> trans_sents;
	//cerr << "TEST SIZE " << test_sents.size() << endl;
  for(int i = 0; i < test_sents.size(); ++i) {
  	//for (int j = 0; j < test_sents[i].size(); ++j) {
  	//	cerr << sourceD.convert(test_sents[i][j]) << " ";
  	//}
  	//cerr << "\n";
  	ComputationGraph cg;
  	Expression encoding = tr.Encoder(cg, test_sents[i]);
  	Expression enc2dec = parameter(cg, tr.p_enc2dec);
  	Expression decbias = parameter(cg, tr.p_decbias);
  	Expression outbias = parameter(cg, tr.p_outbias);
  	Expression enc2out = parameter(cg, tr.p_enc2out);
  	Expression c0 = affine_transform({decbias, enc2dec, encoding});
  	vector<Expression> init(OUT_LAYERS * 2);
  	for (unsigned i = 0; i < OUT_LAYERS; ++i) {
  	      init[i] = pickrange(c0, i * HIDDEN_DIM, i * HIDDEN_DIM + HIDDEN_DIM);
  	      init[i + OUT_LAYERS] = tanh(init[i]);
  	    }
  	tr.outbuilder.new_graph(cg);
  	tr.outbuilder.start_new_sequence(init);
	tr.outbuilder.add_input(lookup(cg, tr.p_c, kSOS2));
  	Expression h_t = tr.outbuilder.back();
  	//cerr << "<s> ";
  	int translation = -1;
  	int count = 0;
	string trans_sent;
  	while(translation != kEOS2 and count < 50){
	  	//cerr << translation << " ";
	    	count++;
	  	Expression u_t = affine_transform({outbias, enc2out, h_t});
	  	Expression probs_embedding = log_softmax(u_t);
	  	vector<float> probs = as_vector(cg.incremental_forward(probs_embedding));
	  	vector<pair<float, int>> prob_idx; // probabilities and indices
	  	for (int k = 0; k < probs.size(); ++k) {
	  	  prob_idx.push_back(make_pair(probs[k], k));
		  //cout << probs[k] << " " << k << "\n";
	  	}
	  	sort(prob_idx.begin(), prob_idx.end(), comparator);
	  	translation = prob_idx[0].second;
	  	//cerr << targetD.convert(translation) << " " << prob_idx[0].first << " " << prob_idx[0].second << "\n";
	  	trans_sent.append(targetD.convert(translation));
	  	trans_sent.append(" ");
	  	Expression x_t = lookup(cg, tr.p_c, translation);
	  	h_t = tr.outbuilder.add_input(x_t);
 	 }
	//cerr << "TRANS SENT " << trans_sent << endl;
	trans_sents.push_back(trans_sent);
  }
  cerr << "\n"; 
	return trans_sents; // vector of strings
}

int main(int argc, char** argv) {
	dynet::initialize(argc, argv);
	bool isTrain = true;
	if (argc == 9){
	  if (!strcmp(argv[8], "-t")) {
	    isTrain = false;
	  }
	}
	if (argc != 5 && argc != 6 && argc != 7 && argc != 8) {
	  cerr << "Usage: " << argv[0] << " train.source train.target dev.source dev.target [test.source] output.model [input.model] [-t]\n";
	  return 1;
	}
	if (isTrain) {
		Model model;
		kSOS = sourceD.convert("<s>");
		kEOS = sourceD.convert("</s>");
		kSOS2 = targetD.convert("<s>");
		kEOS2 = targetD.convert("</s>");
		vector<vector<int>> train_source, train_target;
		vector<vector<int>> dev_source, dev_target;
		vector<vector<int>> test_source;
		string line;
		{
		  ifstream in(argv[1]);
		  assert(in);
		  vector<int> x;
		  string line;
		  while(getline(in, line)) {
		    x = read_sentence(line, &sourceD);
		    train_source.push_back(x);
		  }
		}
		sourceD.freeze();
                sourceD.set_unk("<UNK>");
		{
		  ifstream in(argv[2]);
		  assert(in);
		  vector<int> x;
		  string line;
		  while(getline(in, line)) {
		    x = read_sentence(line, &targetD);
		    train_target.push_back(x);
		  }
		}
		targetD.freeze();
                targetD.set_unk("<UNK>");
		VOCAB_SIZE = targetD.size() + sourceD.size();
		ofstream out("targetDict");
		boost::archive::text_oarchive oa(out);
		oa << targetD;
		ofstream out2("sourceDict");
		boost::archive::text_oarchive oa2(out2);
		oa2 << sourceD;
		{
		  ifstream in(argv[3]);
		  assert(in);
		  vector<int> x;
		  string line;
		  while(getline(in, line)) {
		    x = read_sentence(line, &sourceD);
		    dev_source.push_back(x);
		  }
		}
		{
		  ifstream in(argv[4]);
		  assert(in);
		  vector<int> x;
		  string line;
		  while(getline(in, line)) {
		    x = read_sentence(line, &targetD);
		    dev_target.push_back(x);
		  }
		}	
		{
		  ifstream in(argv[5]);
		  assert(in);
		  vector<int> x;
		  string line;
		  while(getline(in, line)) {
		    x = read_sentence(line, &sourceD);
		    test_source.push_back(x);
		  }
		}
                if (argc >= 8) {
                  cerr << "Reading parameters from " << argv[7] << "...\n";
                  ifstream in(argv[7]);
                  assert(in);
		  boost::archive::text_iarchive ia(in);
        	  ia >> model;
                }	
		Trainer* sgd = new SimpleSGDTrainer(&model);
		sgd->eta_decay = 0.08;
		EncoderDecoder<LSTMBuilder> lm(model);
		double best = 9e+99;
		unsigned report_every_i = 500 ;
	    unsigned dev_every_i_reports = 25;
	    unsigned si = train_source.size();
	    vector<unsigned> order(train_source.size());
	    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
	    bool first = true;
	    int report = 0;
	    double tsize = 0;
	    while(1) {
	          //Timer iteration("completed in");
	          double loss = 0;
	          unsigned ttags = 0;
	          for (unsigned i = 0; i < report_every_i; ++i) {
	            if (si == train_source.size()) {
	              si = 0;
	              if (first) { first = false; } else { sgd->update_epoch(); }
	              cerr << "**SHUFFLE\n";
	              shuffle(order.begin(), order.end(), *rndeng);
	            }
	    		ComputationGraph cg;
	    		const auto& source_sent = train_source[order[si]];
	    		const auto& target_sent = train_target[order[si]];
	    		++si;
	    		ttags += target_sent.size();
	    		tsize += 1;
	    		Expression loss_expr = lm.BuildGraph(source_sent, target_sent, cg);
	    		loss += as_scalar(cg.incremental_forward(loss_expr));
				cg.backward(loss_expr);
				sgd->update(1.0);
			}
			sgd->status();
			cerr << " E = " << (loss / ttags) << " ppl=" << exp(loss / ttags) << "\n";
			report++;
			if (report % dev_every_i_reports == 0) {
			       double dloss = 0;
			       unsigned dtags = 0;
			       double dcorr = 0;
			       for (unsigned di = 0; di < dev_source.size(); ++di) {
			         const auto& x = dev_source[di];
			         const auto& y = dev_target[di];
			         ComputationGraph cg;
			         Expression dloss_expr = lm.BuildGraph(x, y, cg);
			         dloss += as_scalar(cg.incremental_forward(dloss_expr));
			         dtags += y.size();
			       }
			       if (dloss < best) {
			         cerr << endl;
			         cerr << "Current dev performance exceeds previous best, saving model" << endl;
			         best = dloss;
			         ofstream out("EDmodel");
			         boost::archive::binary_oarchive oa(out);
			         oa << model;

					// Run through test set and print translation
					ofstream log;
					std::time_t now = std::time(NULL);
					std::tm * ptm = std::localtime(&now);
					char fname[32];
					std::strftime(fname, 32, "testout_%Y-%m-%dT%H-%M-%S.txt", ptm);  
   					 log.open(fname);
   					 if (log.fail())
   					     perror(fname);

					// Run through test set, printing
					vector<string> trans_sents = Translate(test_source, lm);
					//ostringstream s;
					//copy(trans_sents.begin(), trans_sents.end(), ostream_iterator<string>(s, " "));
					copy(trans_sents.begin(), trans_sents.end(), ostream_iterator<string>(log, "\n"));

					//log << s.str() << "\n";
					log.close();
					cerr << "Printed test output to " << fname << endl;
			    //   for (unsigned di = 0; di < test_source.size(); ++di){
				//       log << Translate(test_source[di], lm) << "\n";
   				//	 log.close();
			       //}

			       for (unsigned di = 0; di < 10; ++di){
				       vector<vector<int>> test;
				       test.push_back(dev_source[di]);
				       vector<string> dev_trans = Translate(test, lm);
						copy(dev_trans.begin(), dev_trans.end(), ostream_iterator<string>(cerr));
						cerr << "\n";
			
				       for (unsigned ti = 0; ti < dev_target[di].size(); ++ti){
							cerr << targetD.convert(dev_target[di][ti]) << " ";
						
					}
					cerr << "\n=============\n";
				}

			       cerr << "***DEV [epoch=" << (tsize / (double)train_source.size()) << "] E = " << (dloss / dtags) << " ppl=" << exp(dloss / dtags) << "\n" << ' ';
			     }
			 }
		}
	}
	 else {
	 cerr << "Test 1";
	 ifstream in3("targetDict");
	 assert(in3);
	 boost::archive::text_iarchive ia3(in3);
	 ia3 >> targetD;

	 cerr << "Test 2";
	 Model model;
	 ifstream in("sourceDict");
	 assert(in);
	 boost::archive::text_iarchive ia(in);
	 ia >> sourceD;



	 sourceD.freeze();
	 targetD.freeze();
	 VOCAB_SIZE = targetD.size();

	 vector<vector<int>> test_source;

	 ifstream in2("EDmodel");
	 assert(in2);
	 boost::archive::binary_iarchive ia2(in2);
	 ia2 >> model;
	 {
	   ifstream in(argv[5]);
	   assert(in);
	   vector<int> x;
	   string line;
	   while(getline(in, line)) {
	     x = read_sentence(line, &sourceD);
	     test_source.push_back(x);
	   }
	 }
	 EncoderDecoder<LSTMBuilder> tr(model);
	 Translate(test_source, tr);
	 }
}

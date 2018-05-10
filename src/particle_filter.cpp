/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;

	// initialize a random engine (?)
	default_random_engine gen;

	// This line creates a normal (Gaussian) distribution for x.
	// with some mean and standard deviation
	std::normal_distribution<double> dist_x(0, std[0]);
	std::normal_distribution<double> dist_y(0, std[1]);
	std::normal_distribution<double> dist_theta(0, std[2]);

	for(int i = 0; i<num_particles; i++){
		Particle p = {};
		p.id = 0;
		p.x = x + dist_x(gen);
		p.y = y + dist_y(gen);
		p.theta = theta + dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
		weights.push_back(1.0);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	// This line creates a normal (Gaussian) distribution for x.
	// with some mean and standard deviation
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);

	for(int i = 0; i < num_particles; i++){
        Particle& p = particles[i]; 
        
        double new_x;
		double new_y;
		double new_theta;

        if(fabs(yaw_rate) < 0.001){
            new_x = p.x + velocity*delta_t*cos(p.theta);
            new_y = p.y + velocity*delta_t*sin(p.theta);
            new_theta = p.theta;
        }else{
            new_x = p.x + (velocity * (sin(p.theta + yaw_rate*delta_t)-sin(p.theta))/yaw_rate);
            new_y = p.y + (velocity * (cos(p.theta)-cos(p.theta + yaw_rate*delta_t))/yaw_rate);
            new_theta = p.theta + yaw_rate*delta_t;
        }

        p.x = new_x + dist_x(gen);
        p.y = new_y + dist_y(gen);
        p.theta = new_theta + dist_theta(gen);
		p.weight = 1.0;
		weights[i] = 1.0;
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(unsigned int i = 0; i < observations.size(); i++){
		auto& obs = observations[i];
		int temp_id;
		double prev_dist = std::numeric_limits<double>::infinity();
		for(int j = 0; j<  predicted.size(); j++){
			double current_dist = sqrt(pow(obs.x-predicted[j].x,2)+pow(obs.y-predicted[j].y,2));
			if(prev_dist > current_dist){
				temp_id = j;
				prev_dist = current_dist;
			}
		}
		obs.id= temp_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for(int i = 0; i < num_particles; i++){
		std::vector<LandmarkObs> trasformed_obs;
		std::vector<LandmarkObs> predictions;
		Particle& particle = particles[i]; 
		for(auto obs: observations){
			LandmarkObs temp_mapped = {};
			temp_mapped.x = particle.x + (cos(particle.theta) * obs.x) - (sin(particle.theta) * obs.y);
			temp_mapped.y = particle.y + (sin(particle.theta) * obs.x) + (cos(particle.theta) * obs.y);
			trasformed_obs.push_back(temp_mapped);
		}

		for(auto landmark: map_landmarks.landmark_list){
			if(fabs(particle.x-landmark.x_f) <= sensor_range && fabs(particle.y-landmark.y_f) <= sensor_range){
				LandmarkObs temp_predict = {};
				temp_predict.x = landmark.x_f;
				temp_predict.y = landmark.y_f;
				temp_predict.id = landmark.id_i;
				predictions.push_back(temp_predict);
			}
		}
		dataAssociation(predictions, trasformed_obs);
		for(auto obs: trasformed_obs){
			double obs_w = ( 1/(2*M_PI*std_landmark[0]*std_landmark[1])) * exp( -( pow(predictions[obs.id].x-obs.x,2)/(2*pow(std_landmark[0], 2)) + (pow(predictions[obs.id].y-obs.y,2)/(2*pow(std_landmark[1], 2))) ) );
			particle.weight *= obs_w;
			weights[i] *=  obs_w;
		}
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<Particle> new_particles;

    random_device rd;
     mt19937 gen(rd());
     discrete_distribution<> distribution(weights.begin(), weights.end());

    for(int i = 0; i < num_particles; i++){
        Particle p = particles[distribution(gen)];
        new_particles.push_back(p);
    }
    particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

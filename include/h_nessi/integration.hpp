
/**
 * @file integration.hpp
 * @brief Quadrature and integration utilities used by time-stepping and convolution routines.
 *
 * This header provides routines to construct polynomial interpolation/differentiation/integration tables,
 * Gregory weights, and the Integrator helper class which exposes commonly used weights and tables for
 * the time-stepping algorithms.
 */

#ifndef INTEGRATION_DECL
#define INTEGRATION_DECL

#include <algorithm>
#include <cmath>
#include <iostream>
#include <complex>
#include <stdlib.h>
#include <Eigen/Eigen>

#include "utils.hpp"
#include "checks.hpp"
#include <vector>

/**
 * @namespace Integration
 * @brief Helper functions and classes for numerical integration and quadrature rules.
 */
namespace Integration {

/**
 * @brief Build polynomial interpolation table for order k.
 * @param k Polynomial order.
 * @param P Output array of size (k+1)*(k+1) populated with interpolation coefficients.
 */
 	void make_poly_interp(int k, std::vector<long double> &P);

/**
 * @brief Build polynomial differentiation table for order k.
 * @param k Polynomial order.
 * @param P Input array of size (k+1)*(k+1) populated with interpolation coefficients.
 * @param D Output array of size (k+1)*(k+1) populated with differentiation coefficients.
 */
 	void make_poly_diff(int k, const std::vector<long double> &P, std::vector<long double> &D);

/**
 * @brief Build polynomial integration table for order k.
 * @param k Polynomial order.
 * @param P Input array of size (k+1)*(k+1) populated with interpolation coefficients.
 * @param I Output array of size (k+1)*(k+1)*(k+1) populated with Integration coefficients.
 */
 	void make_poly_integ(int k, const std::vector<long double> &P, std::vector<long double> &I);

/**
 * @brief Build polynomial interpolation table for order k.
 * @param k Polynomial order.
 * @param P Output array of size (k+1)*(k+1) populated with interpolation coefficients.
 */
 	void make_poly_interp(int k, std::vector<double> &P);

/**
 * @brief Build polynomial differentiation table for order k.
 * @param k Polynomial order.
 * @param D Output array of size (k+1)*(k+1) populated with differentiation coefficients.
 */
 	void make_poly_diff(int k, std::vector<double> &D);

/**
 * @brief Build polynomial integration table for order k.
 * @param k Polynomial order.
 * @param I Output array of size (k+1)*(k+1)*(k+1) populated with Integration coefficients.
 */
 	void make_poly_integ(int k, std::vector<double> &I);

/**
 * @brief Construct Backwards differentiation weights for order k+1.
 * @param k Polynomial order.
 * @param BD Output array of size k+2 populated with backwards differentiation weights.
 */
 	void make_bd_weights(int k, std::vector<double> &BD);

/**
 * @brief Construct starting integration weights using polynomial integration.  
 * @param k Order parameter.
 * @param s Output array of size (k+1)*(k+1).
 */
 	void make_start(int k, std::vector<double> &s);

/**
 * @brief Construct Omega integration weights from Gregory integration.
 * @param k Order parameter.
 * @param O Output array of size (k+1)*(k+1).
 */
	 void make_Omega(int k, std::vector<double> &O);

/**
 * @brief Construct extrapolation weights for order k.
 * @param k Order parameter.
 * @param E Output array of size k+1.
 */
	 void make_ex_weights(int k, std::vector<double> &E); 

/**
 * @class Integrator
 * @brief Precomputes and exposes integration weights and polynomial tables for a chosen order k.
 *
 * The Integrator stores interpolation, differentiation and integration polynomial tables, Gregory weights,
 * boundary-difference weights and extrapolation weights for a fixed order k. These tables are used by
 * the time-stepping and convolution routines in the codebase.
 */
	class Integrator{
		public:

		/**
		 * @brief Construct an Integrator for order k.
		 * @param k Order used to compute tables and weights (0 <= k <= h_nessi::KMAX = 6).
		 *
		 * The constructor allocates and populates internal tables. It throws h_nessi::precondition_error if k is
		 * out of range.
		 */

		Integrator(int k){
			int k1=k+1;
			// IN-1 / LX-17 (tier 2): 0 <= k <= KMAX; throws h_nessi::precondition_error (was: print + abort for k < 0 only)
			HNESSI_REQUIRE(0 <= k && k <= h_nessi::KMAX, "Integration::Integrator::Integrator", HNESSI_KV(k), HNESSI_KV(h_nessi::KMAX));
			poly_interp_.resize(k1*k1);
			poly_diff_.resize(k1*k1);
			poly_integ_.resize(k1*k1*k1);
			bd_weights_.resize(k1+1);
			ex_weights_.resize(k+1);
			gregory_start_.resize(k1*k1);
			gregory_Omega_.resize(k1*k1);
			gregory_omega_offset_ = k*(k1);
			k_=k;
			Integration::make_poly_interp(k_, poly_interp_);
			Integration::make_poly_diff(k_, poly_diff_);
			Integration::make_poly_integ(k_, poly_integ_);
			Integration::make_bd_weights(k_, bd_weights_);
			Integration::make_start(k_, gregory_start_);
			Integration::make_Omega(k_, gregory_Omega_);
			Integration::make_ex_weights(k_, ex_weights_);
			check_weights_();
		}

		/** @brief Return stored order k. */
		int k(void) const {return k_;}
		int get_k(void) const {return k_;}

		/**
		 * @brief Access polynomial interpolation coefficient P[a,l].
		 * @param a First index (0..k)
		 * @param l Second index (0..k)
		 * @return Coefficient value.
		 */
		 double poly_interp(int a, int l) const {
			HNESSI_ASSERT(0 <= a && a <= k_ && 0 <= l && l <= k_, "Integration::Integrator::poly_interp", HNESSI_KV(a), HNESSI_KV(l), HNESSI_KV(k_));  // IN-3, LX-16
			return poly_interp_[a*(k_+1)+l];
	    }
		/**
		 * @brief Access polynomial differentiation coefficient D[m,l].
		 */
			double poly_diff(int m, int l) const {
			HNESSI_ASSERT(0 <= m && m <= k_ && 0 <= l && l <= k_, "Integration::Integrator::poly_diff", HNESSI_KV(m), HNESSI_KV(l), HNESSI_KV(k_));  // IN-4
			return poly_diff_[m*(k_+1)+l];
	    }
		/**
		 * @brief Access polynomial integration coefficient I[m,n,l].
		 */
		double poly_integ(int m, int n, int l) const {
			HNESSI_ASSERT(0 <= m && m <= k_ && 0 <= n && n <= k_ && 0 <= l && l <= k_, "Integration::Integrator::poly_integ", HNESSI_KV(m), HNESSI_KV(n),
			              HNESSI_KV(l), HNESSI_KV(k_));  // IN-5
			return poly_integ_[m*(k_+1)*(k_+1)+n*(k_+1)+l];
		}
		/**
		 * @brief Backwards differentiation weight accessor.
		 * @param l index
		 * @return weight value
		 */
		double bd_weights(int l) const {
			HNESSI_ASSERT(0 <= l && l <= k_+1, "Integration::Integrator::bd_weights", HNESSI_KV(l), HNESSI_KV(k_));  // IN-6
			return bd_weights_[l];
		}
		/**
		 * @brief Gregory weights g[n,j] used for integration.
		 * @param n first index
		 * @param j second index
		 * @return weight value
		 */
		double gregory_weights(int n, int j) const {
			// IN-7 (tier 1): a negative index or j > max(n, k) reads outside the tables (for n <= k the start weights
			// of the integral 0..n cover all k+1 interpolation points, so j <= k is valid there)
			HNESSI_ASSERT(0 <= n && 0 <= j && j <= std::max(n, k_), "Integration::Integrator::gregory_weights", HNESSI_KV(n),
			              HNESSI_KV(j), HNESSI_KV(k_));
			if(n<=k_&&j<=k_){
				return gregory_start_[n*(k_+1)+j];
			}
			else if(n<=2*k_+1){
				if(j<=k_) return gregory_Omega_[(n-k_-1)*(k_+1)+j];
					else return gregory_Omega_[gregory_omega_offset_ + (n-j)];
			}
			else{
				if(j<=k_) return gregory_Omega_[gregory_omega_offset_ + j];
				else if(j<n-k_) return 1;
				else return gregory_Omega_[gregory_omega_offset_ + (n-j)];
			}
		}
		/** @brief Access boundary correction weights used in Gregory integration. */
		double omega(int j) const {
			HNESSI_ASSERT(0 <= j && j <= k_, "Integration::Integrator::omega", HNESSI_KV(j), HNESSI_KV(k_));  // IN-8
			return gregory_Omega_[gregory_omega_offset_ + j];
		}
		/** @brief Start integration weights table accessor. */
		double start(int i, int j) const {
			HNESSI_ASSERT(0 <= i && i <= k_ && 0 <= j && j <= k_, "Integration::Integrator::start", HNESSI_KV(i), HNESSI_KV(j), HNESSI_KV(k_));  // IN-9
			return gregory_start_[i*(k_+1)+j];
		}
		/** @brief Omega integration weights accessor. (boundary correction regions overlap)*/
		double Omega(int i, int j) const {
			HNESSI_ASSERT(0 <= i && i <= k_ && 0 <= j && j <= k_, "Integration::Integrator::Omega", HNESSI_KV(i), HNESSI_KV(j), HNESSI_KV(k_));  // IN-9
			return gregory_Omega_[i*(k_+1)+j];
		}
		/** @brief Extrapolation weights accessor. */
		double ex_weights(int j) const {
			HNESSI_ASSERT(0 <= j && j <= k_, "Integration::Integrator::ex_weights", HNESSI_KV(j), HNESSI_KV(k_));  // IN-8
			return ex_weights_[j];
		}

		private:
			// IN-2 (tier 2, once per construction): every weight finite; the backward-difference weights differentiate
			// a constant to zero and the extrapolation weights reproduce a constant (detects an ill-conditioned
			// long-double solve for large k, ISSUES P3)
			void check_weights_() const {
				auto finite = [](const std::vector<double> &v) {
					for(double x : v) if(!std::isfinite(x)) return false;
					return true;
				};
				bool all_finite = finite(poly_interp_) && finite(poly_diff_) && finite(poly_integ_) && finite(bd_weights_) &&
				                  finite(ex_weights_) && finite(gregory_start_) && finite(gregory_Omega_);
				HNESSI_ENSURE(all_finite, "Integration::Integrator::Integrator", HNESSI_KV(k_));
				double sbd = 0, sex = 0;
				for(double x : bd_weights_) sbd += x;
				for(double x : ex_weights_) sex += x;
				HNESSI_ENSURE(std::abs(sbd) < 1e-10 && std::abs(sex - 1) < 1e-10, "Integration::Integrator::Integrator",
				              HNESSI_KV(k_), HNESSI_KV(sbd), HNESSI_KV(sex - 1),
				              h_nessi::detail::note{"the integration weights are inaccurate (ill-conditioned for this k)"});
			}
			int k_;
			std::vector<double> poly_interp_;
			std::vector<double> poly_diff_;
			std::vector<double> poly_integ_;
			std::vector<double> bd_weights_;
		    std::vector<double> ex_weights_;

			std::vector<double> gregory_start_;
			std::vector<double> gregory_Omega_;
			int gregory_omega_offset_;
		};


}//Namespace

#endif



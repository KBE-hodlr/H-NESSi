#include "observables.hpp"

namespace observables {
   std::array<double,4> get_obs_local
   (int tstp,
    int Nk_rank,
    lattice_2d_ysymm &lattice,
    std::vector<int> &kindex_rank,
    std::vector<std::unique_ptr<kpoint>> &corrK_rank
   )
   {
      int kindex;
      h_nessi::ZMatrix rtmp(1,1),etmp(1,1),vkxtmp, vkytmp;
      double jx=0.0,jy=0.0,Ekin=0.0,Num=0.0;

      double my_density_k;
      
      for(int kindex=0; kindex<Nk_rank; kindex++){
         h_nessi::DMatrix dtmp(1,1);
         dtmp = corrK_rank[kindex]->rho_[tstp];
         my_density_k = std::real(dtmp(0,0));
         
         int kxi = kindex_rank[kindex]/(lattice.L_/2+1);
         int kyi = kindex_rank[kindex]%(lattice.L_/2+1);

         double kx = lattice.kpoints_[kindex_rank[kindex]].x();
         double ky = lattice.kpoints_[kindex_rank[kindex]].y();

         rtmp(0,0) = my_density_k;
         lattice.vk(vkxtmp,vkytmp,tstp,lattice.kpoints_[kindex_rank[kindex]]);
         
         jx+=std::real((vkxtmp*rtmp).trace())/(lattice.L_*lattice.L_);
         jy+=std::real((vkytmp*rtmp).trace())/(lattice.L_*lattice.L_);
         
         etmp(0,0) = -lattice.mu_-2*lattice.J_*(std::cos(kx)+std::cos(ky));   
         Ekin+=std::real((rtmp*etmp).trace())/(lattice.L_*lattice.L_);
         Num+=std::real((rtmp).trace())/(lattice.L_*lattice.L_);

         if(kyi!=lattice.L_/2 && kyi!=0){
            jx+=std::real((vkxtmp*rtmp).trace())/(lattice.L_*lattice.L_);
            jy-=std::real((vkytmp*rtmp).trace())/(lattice.L_*lattice.L_);
            Ekin+=std::real((rtmp*etmp).trace())/(lattice.L_*lattice.L_);  
            Num+=std::real((rtmp).trace())/(lattice.L_*lattice.L_);
         }
      }

      std::array<double,4> obs = {2.0*jx,2.0*jy,2.0*Ekin,Num};

      return obs;

   }
   
   double bz_weight(int global_k, lattice_2d_ysymm &lattice)
   {
      int kyi = global_k%(lattice.L_/2+1);
      double w = 1.0/(lattice.L_*lattice.L_);
      return (kyi!=lattice.L_/2 && kyi!=0) ? 2.0*w : w;
   }

   double get_energy_band_local
   (int tstp,
    int Nk_rank,
    lattice_2d_ysymm &lattice,
    std::vector<int> &kindex_rank,
    std::vector<std::unique_ptr<kpoint>> &corrK_rank
   )
   {
      double Eband = 0.0;
      for(int kindex=0; kindex<Nk_rank; kindex++){
         int global_k = kindex_rank[kindex];
         double rho_k = corrK_rank[kindex]->rho_[tstp](0,0);   // -i G^<_k(t,t), per spin
         Eband += bz_weight(global_k, lattice) * lattice.epsk_band(tstp, lattice.kpoints_[global_k]) * rho_k;
      }
      return 2.0*Eband;   // two spins
   }

   std::array<double,4> get_obs_local_no_parallel(int tstp,
                        lattice_2d_ysymm &lattice,
                        std::vector<std::unique_ptr<kpoint>> &corrK_rank
                       )
   {
      h_nessi::ZMatrix rtmp(1,1),etmp(1,1),vkxtmp, vkytmp;
      double jx=0.0,jy=0.0,Ekin=0.0,Num=0.0;

      double my_density_k;
      
      for(int ki=0; ki<lattice.Nk_; ki++){
         h_nessi::DMatrix dtmp(1,1);
         dtmp = corrK_rank[ki]->rho_[tstp];
         my_density_k = std::real(dtmp(0,0));
         
         int kxi = ki/(lattice.L_/2+1);
         int kyi = ki%(lattice.L_/2+1);

         double kx = lattice.kpoints_[ki].x();
         double ky = lattice.kpoints_[ki].y();

         rtmp(0,0) = my_density_k;
         lattice.vk(vkxtmp,vkytmp,tstp,lattice.kpoints_[ki]);
         
         jx+=std::real((vkxtmp*rtmp).trace())/(lattice.L_*lattice.L_);
         jy+=std::real((vkytmp*rtmp).trace())/(lattice.L_*lattice.L_);
         
         etmp(0,0) = -lattice.mu_-2*lattice.J_*(std::cos(kx)+std::cos(ky));   
         Ekin+=std::real((rtmp*etmp).trace())/(lattice.L_*lattice.L_);
         Num+=std::real((rtmp).trace())/(lattice.L_*lattice.L_);

         if(kyi!=lattice.L_/2 && kyi!=0){
            jx+=std::real((vkxtmp*rtmp).trace())/(lattice.L_*lattice.L_);
            jy-=std::real((vkytmp*rtmp).trace())/(lattice.L_*lattice.L_);
            Ekin+=std::real((rtmp*etmp).trace())/(lattice.L_*lattice.L_);  
            Num+=std::real((rtmp).trace())/(lattice.L_*lattice.L_);
         }
      }

      std::array<double,4> obs = {2.0*jx,2.0*jy,2.0*Ekin,Num};

      return obs;

   }
}


#include <iostream>
#include <string>
#include <vector>
#include <complex>
#include "extra_tools.h"
#include "tester.h"
#include <fftw3.h>
#include <boost/math/statistics/linear_regression.hpp>

using namespace std;

vector<vector<int>> super_argsort(vector<vector<std::complex<double>>>& data){
    vector<vector<int>> res(data.size(), vector<int>(data[0].size()));
    vector<double> temp_data(data.size());
    vector<int> temp_index(data.size());
    for(size_t j = 0; j < data[0].size();++j){
        for(size_t i = 0; i < data.size();++i){
            temp_data[i] = abs(data[i][j]);
        }
        temp_index = argsort(temp_data);
        for(size_t i = 0; i < data.size();++i){
            res[i][j] = temp_index[i];
        }
    }
    return res;
}

vector<vector<complex<double>>> Azimuth_diff(vector<vector<complex<double>>> &g){
    vector<vector<complex<double>>> temp;
    for(size_t i = 0; i < g.size() - 1;i++){
        temp.push_back(g[i+1] - g[i]);
    }
    return temp;
}

void Azimuth_FFT(vector<vector<std::complex<double>>> &Raw_data, size_t size_range, size_t size_azimuth, int sign){
    fftw_plan plan_f;
    vector<complex<double>> temp(size_azimuth);

    shift(Raw_data);

    for(size_t j = 0; j < size_range;j++){
        for(size_t i = 0; i < size_azimuth;i++){
            temp[i] = Raw_data[i][j];
        }
        //plan_f = fftw_plan_dft_1d(size_azimuth, (fftw_complex*) &temp[0],
        //                          (fftw_complex*) &temp[0], FFTW_FORWARD, FFTW_ESTIMATE); //было
        plan_f = fftw_plan_dft_1d(size_azimuth, (fftw_complex*) &temp[0],
                                  (fftw_complex*) &temp[0], sign, FFTW_ESTIMATE); //сделано на основе RITSAR
        fftw_execute(plan_f);
        for(size_t i = 0; i < size_azimuth;i++){
            Raw_data[i][j] = complex<double>(temp[i].real()/size_azimuth, temp[i].imag()/size_azimuth);//https://habr.com/ru/company/otus/blog/449996/
        }
    }
    shift(Raw_data);
    fftw_destroy_plan(plan_f);
}

void PGA(vector<vector<complex<double>>>& data, string win = "auto", vector<double> win_params = {100.0, 0.5}){
    //Предусматривается, что матрица данных data имеет квадратный вид
    //Derive parameters
    size_t npulses = data.size();
    size_t nsamples = data[0].size();

    //Initialize loop variables
    size_t max_iter = 30;
    std::vector<double> af_ph(npulses, 0.0);
    vector<vector<int>> indexes(data.size(), vector<int>(data[0].size()));
    vector<double> rms;//потом можно будет заменить на 1 переменную(возможно)
    //Compute phase error and apply correction
    for(size_t iii = 0; iii < max_iter; ++iii){
        //Find brightest azimuth sample in each range bin
        indexes = super_argsort(data);
        vector<int> index = indexes[npulses - 1];//index = np.argsort(np.abs(img_af), axis=0)[-1]

        //Circularly shift image so max values line up

        vector<vector<complex<double>>> f(npulses, vector<complex<double>>(nsamples));
        for(size_t i = 0; i < npulses;++i){
            for(size_t j = 0; j < nsamples; ++j){
                f[i][j] = data[i][(j+npulses/2-index[j])%nsamples];
            }
        }
        vector<double> window;
        if(win == "auto"){
            //Compute window width
            vector<double> s(nsamples);
            for(size_t i = 0; i < npulses;++i){
                for(size_t j = 0; j < nsamples;++j){
                    s[j] = s[j] + abs(f[i][j]) * abs(f[i][j]);
                }
            }
            double s_max = abs(*max(s.begin(), s.end()));
            double width = 0.0;
            for(size_t i = 0; i < nsamples;++i){
                s[i] = 10.0 * log10(s[i]/s_max);
                if(s[i] > -30.0){
                    width += s[i];
                }
            }

            window = fill_up(npulses/2-width/2,npulses/2+width/2);
        }else{
            //Compute window width using win_params if win not set to 'auto'
            int width = (int)pow(win_params[0]*win_params[1], iii);
            window = fill_up(npulses/2-width/2,npulses/2+width/2);
            if(width < 5) break;
        }
        //Window image
        vector<vector<complex<double>>> g(npulses, vector<complex<double>>(nsamples));
        for(auto a: window){
            g[a] = f[a];
        }

        //Fourier Transform
        Azimuth_FFT(g,nsamples,npulses, FFTW_BACKWARD);




        //take derivative
        std::vector<vector<complex<double>>> G_dot = Azimuth_diff(g);
        G_dot.push_back(G_dot[G_dot.size() - 1]);

        //Estimate Spectrum for the derivative of the phase error
        //Integrate to obtain estimate of phase error(Jak)
        std::vector<double> phi(npulses, 0.0);
        for(size_t i = 0; i < npulses;++i){
            double sum_1 = 0, sum_2 = 0;
            for(size_t j = 0; j < nsamples; ++j){
                sum_1 += imag(conj(g[i][j])*G_dot[i][j]);
                sum_2 += abs(g[i][j]) * abs(g[i][j]);
            }
            if(i == 0){
                phi[i] = sum_1/sum_2;
            }else {
                phi[i] = phi[i - 1] + (sum_1 / sum_2);
            }
        }



        //Remove linear trend
        vector<double> t = fill_up(0,nsamples);
        using boost::math::statistics::simple_ordinary_least_squares_with_R_squared;

        auto [intercept, slope, R] = simple_ordinary_least_squares_with_R_squared(t, phi);

        double line;
        double mean = 0.0;
        for(size_t i = 0; i < nsamples;++i){
            line = slope*t[i]+intercept;
            phi[i] -= line;
            mean += phi[i]*phi[i];
        }
        mean = mean/nsamples;
        rms.push_back(mean);
        if(win == "auto"){
            if(rms[iii] < 0.1) break;
        }
        //Apply correction
        std::vector<vector<double>> phi2 = tile_y(phi, {1, static_cast<int>(nsamples)});
        Azimuth_FFT(data,nsamples,npulses, FFTW_BACKWARD);
        for(size_t i = 0; i < npulses; ++i){
            for(size_t j = 0; j < nsamples;++j){
                data[i][j] *= exp(-phi2[i][j]*static_cast<complex<double>>(I));
            }
        }
        Azimuth_FFT(data,nsamples,npulses, FFTW_FORWARD);
        //Store phase
        af_ph = af_ph + phi;
    }
}


int main() {
    vector<vector<std::complex<double>>> Raw_data = read_file(false, "/home/ivanemekeev/CLionProjects/SAR-data/Example_before_autofocusing.txt");
    PGA(Raw_data);
    equality(Raw_data, "/home/ivanemekeev/CLionProjects/SAR-data/Example_after_autofocusing.txt","Autofocusing", "inf");
    return 0;
}

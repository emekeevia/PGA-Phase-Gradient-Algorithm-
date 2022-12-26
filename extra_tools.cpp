#include "extra_tools.h"
#include "CubicSlpine.h"


vector<double> fill_up(const double& start,const double& finish, const double& step){
    vector<double> mass;
    for(double temp = start; temp <= finish;temp+=step){
        mass.push_back(temp);
    }
    return mass;
}

vector<double> real(vector<std::complex<double>>& data){
    vector<double> temp(data.size());
    for(size_t i = 0; i < data.size();i++){
        temp[i] = data[i].real();
    }
    return temp;
}

vector<double> imag(vector<std::complex<double>>& data){
    vector<double> temp(data.size());
    for(size_t i = 0; i < data.size();i++){
        temp[i] = data[i].imag();
    }
    return temp;
}

vector<double> fill_up(const int& start, const int& finish){
    vector<double> mass;
    for(double temp = start; temp < finish;temp++){
        mass.push_back(temp);
    }
    return mass;
}

vector<double> fill_up(const double& start,const double& finish, size_t N){
    vector<double> mass(N);
    double step = (finish - start)/(N-1);
    for(size_t i = 0; i < N;i++){
        mass[i] = start + i*step;
    }
    return mass;
}
vector<double> fill_up(const int& start, const int& finish, int N){
    vector<double> mass;
    double step = (static_cast<double>(finish) - static_cast<double>(start) + 1.0)/static_cast<double>(N);
    for(int i = 0; i < N;i++){
        mass.push_back(static_cast<double>(start + i*step));
    }
    return mass;
}

vector<complex<double>> conjugate(vector<complex<double>>& orig){
    size_t size = orig.size();
    vector<complex<double>> sopr(size);
    for(size_t i = 0; i < size;i++){
        sopr[i] = conj(orig[i]);
    }
    return sopr;
}

vector<complex<double>> operator*(const vector<complex<double>>& v1, const vector<complex<double>>& v2){
    size_t size = v1.size();
    if(size != v2.size()){
        throw("Not equal size!");
    }
    vector<complex<double>> temp(size);
    for(size_t i = 0; i < size;i++){
        temp[i] = v1[i]*v2[i];
    }
    return temp;
}

void shift(vector<vector<std::complex<double>>> &Raw_data){
    size_t az_size = Raw_data.size();
    for(size_t i = 0; i < az_size;i++){
        iFFTshift(Raw_data[i]);
    }
    iFFTshift(Raw_data);
}

size_t partition(std::vector<double>& v, std::vector<int>& temp, size_t l, size_t r){
    double pivot = v[(l + r) / 2];
    size_t i = l;
    size_t j = r;
    while (i <= j){
        while (v[i] < pivot){
            i++;
        }
        while (v[j] > pivot){
            j--;
        }
        if (i >= j){
            break;
        }

        swap(v[i], v[j]);
        swap(temp[i++], temp[j--]);
    }

    return j;
}


void quick_sort(std::vector<double>& v, std::vector<int>& temp, size_t l, size_t r){
    if(l < r){
        size_t q = partition(v,temp, l, r);
        quick_sort(v, temp, l, q);
        quick_sort(v, temp, q + 1, r);
    }

}

void set_array(vector<vector<std::complex<double>>>& array, vector<double> values, vector<int> bound_r, vector<int> bound_c){
    for(size_t i = bound_r[0];i < bound_r[1];i++){
        for(size_t j = bound_c[0]; j < bound_c[1];j++){
            array[i][j] = values[0];
        }
    }
}

void fill_up_KY(vector<double>& KY, vector<double>& KR,double KX_i){
    double temp;
    for(size_t i = 0; i < KR.size();i++){
        temp = KR[i] * KR[i] - (KX_i * KX_i);
        KY[i] = sqrt(temp);
    }
}


std::vector<int> argsort(std::vector<double>& v){
    std::vector<int> temp(v.size(), 0);
    for(int i = 0; i < v.size();i++){
        temp[i] = i;
    }
    quick_sort(v, temp,0, v.size() - 1);
    return temp;
}

void interp1d(std::vector<double>& x,std::vector<double>& y, std::vector<double>& x_new, std::vector<double>& y_new,  string kind){
    std::vector<int> ind = argsort(x);
    std::vector<double> data_sorted(x.size());

    for(int i = 0;i < y.size();i++){
        data_sorted[i] = y[ind[i]];
    }

    std::vector<bool> entry(x_new.size(), true);
    std::vector<int> x_new_indices(x_new.size());
    for(int i = 0;i < x_new.size();i++){
        size_t start = lower_bound(x.begin(), x.end(), x_new[i]) - x.begin();
        if(x_new[i] < x[0]) {
            entry[i] = false;
        }
        if(x_new[i] == x[start]){
            start++;
        }
        if(start > x.size() - 1){
            start = x.size() - 1;
            entry[i] = false;
        }
        x_new_indices[i]=(int)start;
    }

    std::vector<int> lo = x_new_indices - 1;
    std::vector<int> hi = x_new_indices;

    double x_lo , x_hi , y_lo , y_hi ;
    double slope ;


    if(kind == "linear"){
        for(size_t i = 0; i < lo.size();i++){
            x_lo = x[lo[i]];
            x_hi = x[hi[i]];
            y_lo = data_sorted[lo[i]];
            y_hi = data_sorted[hi[i]];

            if(x_hi == x_lo){
                y_new[i] = 0.0;
            }else{
                if(entry[i]){
                    slope = (y_hi - y_lo)/(x_hi - x_lo);
                    y_new[i] = (slope * (x_new[i] - x_lo)) + y_lo;
                }else{
                    y_new[i] = 0.0;
                }
            }
        }
    }else if(kind == "kubic"){
        CubicSpline spline(x, data_sorted);
        for(size_t i = 0; i < lo.size();i++){
            y_new[i] = spline.interpolate(x_new[i]);
        }
    }
}

vector<double> taylor(size_t nsamples, size_t S_L=43){
    vector<double> xi = fill_up(-0.5, 0.5, nsamples);

    double A = (1.0/M_PI) * acosh(pow(10.0, ((double)S_L*1.0)/20.0));
    int n_bar = (int)(2*A*A + 0.5) + 1;
    double sigma_p = (double)n_bar/sqrt(A*A+((double)n_bar-0.5)*((double)n_bar-0.5));
    vector<double> m = fill_up(1, n_bar);
    vector<double> n = fill_up(1,n_bar);
    vector<double> F_m(n_bar-1, 0.0);
    for(auto i: m){
        double num = 1.0;
        double den = 1.0;
        for(auto j: n){
            num = num*pow(-1, i+1)*(1-i*i*1.0/(sigma_p*sigma_p)/(A*A+(j-0.5)*(j-0.5)));
            if(i!=j){
                den = den*(1-i*i*1.0/(j*j));
            }
        }
        F_m[i-1] = num/den;
    }


    vector<double> w(nsamples, 1.0);
    for(auto i : m) {
        for(size_t j = 0;j < nsamples;j++){
            w[j] += F_m[i - 1] * cos(2 * M_PI * i * xi[j]);
        }
    }
    w = w/(*max_element(w.begin(), w.end()));

    return w;
}

vector<vector<complex<double>>> pad_constant(vector<vector<std::complex<double>>>& array,vector<vector<int>> pad_width){

    vector<vector<complex<double>>> new_array(array.size() +pad_width[0][0] + pad_width[0][1], vector<complex<double>>(array[0].size() +pad_width[1][0] + pad_width[1][1], 0.0));
    for(size_t i = 0; i < array.size();i++){
        for(size_t j = 0; j < array[0].size() ;j++){
            new_array[i+pad_width[0][0]][j+pad_width[1][0]] = array[i][j];
        }
    }
    //equality(new_array, "/home/ivanemekeev/CLionProjects/SAR-data/Example_with_rectangle_after_expansion.txt","Stolt interpolation part 3 after expansion", "2");

    vector<vector<double>> values(2, vector<double>(2, 0.0));
    for(size_t k = 0; k < 2;k++){
        vector<vector<complex<double>>> roi;
        if(k == 0){
            for(size_t i = 0;i < new_array.size();i++){
                vector<complex<double>> temp_temp;
                for(size_t j = pad_width[1][0]; j< new_array.size()-pad_width[1][0]; j++){
                    temp_temp.push_back(new_array[i][j]);
                }
                roi.push_back(temp_temp);
            }
        }else{
            roi = new_array;
        }

        if(k == 0){
            //left_slice
            vector<int> rows_l = {0, pad_width[k][0]};
            vector<int> cols_l = {0, (int)roi[0].size()};
            set_array(roi, values[k], rows_l, cols_l);

            //right_slice
            vector<int> rows_r = {(int)roi.size()-pad_width[k][1], (int)roi.size()};
            vector<int> cols_r = {0, (int)roi[0].size()};
            set_array(roi, values[k], rows_r, cols_r);
        }else{
            //left_slice
            vector<int> rows_l = {0, (int)roi.size()};
            vector<int> cols_l = {0, pad_width[k][0]};

            set_array(roi, values[k], rows_l, cols_l);
            //right_slice
            vector<int> rows_r = {0, (int)roi.size()};
            vector<int> cols_r = {(int)roi[0].size() - pad_width[k][1], (int)roi[0].size()};

            set_array(roi, values[k], rows_r, cols_r);

        }


        //k = axis
        //pad_width[k] = width_pair
        //values[k] = value_pair

    }
    return new_array;
}


std::vector<std::vector<std::complex<double>>> read_file(bool pen_writing, string path){
    //Test_Image.mat
    std::vector<std::vector<std::complex<double>>> v;
    std::string line, elem;
    std::vector<std::complex<double>> temp;
    std::complex<double> ch;
    double real, im;
    char sign;
    std::ifstream in_; // окрываем файл для чтения
    in_.open(path, std::ios::in);
    if (in_.is_open()) {
        while (getline(in_, line)) {
            std::stringstream ss, micro_ss;
            ss << line;
            while (ss >> elem) {

                if (pen_writing) {
                    micro_ss << elem;
                    micro_ss >> real;
                    micro_ss.get(sign);
                    micro_ss >> im;
                    if (sign == '-') {
                        im *= -1;
                    }
                    micro_ss.str(std::string());
                    ch = complex<double>(real, im);
                } else {
                    micro_ss << elem;
                    micro_ss >> ch;
                }
                temp.push_back(ch);
            }
            v.push_back(temp);
            temp.clear();
        }
    }
    in_.close();

    return v;
}

void Write_in_file(vector<vector<complex<double>>>& v, string file_name){
    std::ofstream out;          // поток для записи
    out.open(file_name + ".txt");
    size_t azimuth_size = v.size();
    size_t range_size = v[0].size();

    for(size_t i = 0; i < azimuth_size;i++){
        for(size_t j = 0; j < range_size;j++){
            if(j != range_size - 1){
                out << abs(v[i][j]) << ' ';
            }else {
                out << abs(v[i][j]);
            }
        }
        out << '\n';
    }
    out.close();
}







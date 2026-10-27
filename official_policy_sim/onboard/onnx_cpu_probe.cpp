// Offline ONNX CPU inference only: no ROS, network, keyboard or motor access.
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "offline_only");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    Ort::Session session(env, argv[1], options);
    const auto in = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const auto out = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (session.GetInputCount()!=1 || session.GetOutputCount()!=1 || in!=std::vector<int64_t>({1,960}) || out!=std::vector<int64_t>({1,29})) return 3;
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char* inputs[] = {input_name.get()};
    const char* outputs[] = {output_name.get()};
    std::array<float,960> x{};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, x.data(), x.size(), in.data(), in.size());
    std::vector<double> times;
    std::array<float,29> zero_output{};
    for (int n=0;n<520;++n) {
        // Deterministic small test vectors; these are not physical observations.
        for (size_t i=0;i<x.size();++i) x[i]=n==0 ? 0.0f : .1f*std::sin(float(i+n)*.013f);
        const auto start=std::chrono::steady_clock::now();
        auto result=session.Run(Ort::RunOptions{nullptr},inputs,&tensor,1,outputs,1);
        const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        const auto* y=result[0].GetTensorData<float>();
        for(int i=0;i<29;++i) {if(!std::isfinite(y[i])) return 4;if(n==0)zero_output[i]=y[i];}
        if(n>=20)times.push_back(elapsed);
    }
    std::sort(times.begin(),times.end());
    std::cout << "{\"offline_only\":true,\"hardware_commands_sent\":0,\"onnxruntime\":\"" << OrtGetApiBase()->GetVersionString()
              << "\",\"input_shape\":[1,960],\"output_shape\":[1,29],\"timed_samples\":" << times.size()
              << ",\"mean_ms\":" << std::accumulate(times.begin(),times.end(),0.0)/times.size()
              << ",\"p95_ms\":" << times[size_t(.95*(times.size()-1))] << ",\"max_ms\":" << times.back() << ",\"zero_input_output\":[";
    for(int i=0;i<29;++i)std::cout<<(i ? "," : "")<<zero_output[i];
    std::cout << "]}\n";
}

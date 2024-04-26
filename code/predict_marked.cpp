#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <opencv2/opencv.hpp>

int main() {
    // 创建一个解释器
    std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile("path_to_your_model.mnn"));

    // 创建一个会话
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU; // 使用CPU进行推理
    MNN::Session* session = net->createSession(config);

    // 获取输入张量
    MNN::Tensor* input_tensor = net->getSessionInput(session, NULL);

    // 创建一个OpenCV矩阵
    cv::Mat image = cv::imread("path_to_your_image.jpg");

    // 将OpenCV矩阵转换为MNN张量
    // 注意：你可能需要根据你的模型的输入要求对这部分代码进行修改
    cv::resize(image, image, cv::Size(224, 224));
    image.convertTo(image, CV_32FC3);
    image = (image - 127.5) / 127.5;
    std::shared_ptr<MNN::Tensor> tensor(new MNN::Tensor(input_tensor, MNN::Tensor::TENSORFLOW));
    memcpy(tensor->host<float>(), image.data, tensor->size());

    // 将输入数据复制到输入张量
    input_tensor->copyFromHostTensor(tensor.get());

    // 运行模型
    net->runSession(session);

    // 获取输出张量
    MNN::Tensor* output_tensor = net->getSessionOutput(session, NULL);

    // 打印输出
    for (int i = 0; i < output_tensor->elementSize(); i++) {
        std::cout << output_tensor->host<float>()[i] << std::endl;
    }

    return 0;
}
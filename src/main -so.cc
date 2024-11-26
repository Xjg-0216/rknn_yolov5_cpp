#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <boost/filesystem.hpp>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define _BASETSD_H
#define PERF_WITH_POST 1

#include "RgaUtils.h"
#include "postprocess.h"
#include "rknn_api.h"
#include "preprocess.h"

namespace fs = boost::filesystem;
using namespace std;

/*--   Main Functions---------------------------*/
/*--- Functions------------------------------------------*/
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
	std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
	for (int i = 1; i < attr->n_dims; ++i)
	{
		shape_str += ", " + std::to_string(attr->dims[i]);
	}

	printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
		"type=%s, qnt_type=%s, "
		"zp=%d, scale=%f\n",
		attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
		attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
		get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
	unsigned char *data;  int ret;
	data = NULL;
	if (NULL == fp)
	{
		return NULL;
	}

	ret = fseek(fp, ofst, SEEK_SET);
	if (ret != 0)
	{
		printf("blob seek failure.\n");
		return NULL;
	}

	data = (unsigned char *)malloc(sz);
	if (data == NULL)
	{
		printf("buffer malloc failure.\n");
		return NULL;
	}
	ret = fread(data, 1, sz, fp);
	return data;
}

//加载模型文件best+.rknn
static unsigned char *load_model(const char *filename, int *model_size)
{
	FILE *fp;    unsigned char *data;
	fp = fopen(filename, "rb");
	if (NULL == fp)
	{
		printf("Open file %s failed.\n", filename);
		return NULL;
	}

	fseek(fp, 0, SEEK_END);
	int size = ftell(fp);
	data = load_data(fp, 0, size);
	fclose(fp);
	*model_size = size;
	return data;
}

//识别结构结构
struct RKnn
{
	int left;//=int(xyxy[0]);              //#目标左像素坐标
    int top;//=int(xyxy[1]);               //#目标上像素坐标
    int w;//=(int(xyxy[2])-int(xyxy[0]));  //#目标像素宽度
    int h;//=(int(xyxy[3])-int(xyxy[1]));  //#目标像素高度
	//char name[128];				//如果识别成功，就是识别后的图片+方框
};

rknn_context	ctx;
rknn_input		inputs[1];
rknn_input_output_num io_num;
rknn_tensor_attr output_attrs[3];
bool	g_init=false;	//是否已经初始化了，初始值false表示没有初始化rknn识别库

extern "C"{ 
//初始化rknn库，根据置信度阈值，0-1
int initRknn(float fBox_thresh)
{
	//gettimeofday(&start_time, NULL);
	int ret;	
	size_t actual_size = 0;
	int img_width = 0, img_height = 0, img_channel = 0;//图片宽度，高度，通道id
	const float nms_threshold 	= NMS_THRESH;		// 默认的NMS阈值
	float box_conf_threshold	= BOX_THRESH;		// 默认的置信度阈值
	
	char model_name[]={"best-3.rknn"};	// (char *)argv[1];			//输入的model file
	char input_path[128] ={0};			// argv[2];				//输入的识别图片参数
	std::string option 		= "letterbox";
	std::string out_path	= "./out.jpg";     //识别后输出的文件

	box_conf_threshold=fBox_thresh ; //atof(argv[1]);	// 默认的置信度阈值
	//init rga context
	rga_buffer_t src;		rga_buffer_t dst;
	memset(&src, 0, sizeof(src));		memset(&dst, 0, sizeof(dst));

	printf("model name=%s box_conf_threshold = %.2f, nms_threshold = %.2f\n", model_name,box_conf_threshold, nms_threshold);

	/* Create the neural network */
	int model_data_size = 0;
	unsigned char *model_data = load_model(model_name, &model_data_size);
	ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
	if (ret < 0)
	{
		printf("rknn_init error ret=%d\n", ret);
		return -1;
	}

	rknn_sdk_version version;
	ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
	if (ret < 0)
	{
		printf("rknn_init error ret=%d\n", ret);
		return -1;
	}
	printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

	
	ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	if (ret < 0)
	{
		printf("rknn_init error ret=%d\n", ret);
		return -1;
	}
	printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

	rknn_tensor_attr input_attrs[io_num.n_input];
	memset(input_attrs, 0, sizeof(input_attrs));
	for (int i = 0; i < io_num.n_input; i++)
	{
		input_attrs[i].index = i;
		ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
		if (ret < 0)
		{
			printf("rknn_init error ret=%d\n", ret);
			return -1;
		}
		dump_tensor_attr(&(input_attrs[i]));
	}

	//rknn_tensor_attr output_attrs[io_num.n_output];
	memset(output_attrs, 0, sizeof(output_attrs));
	for (int i = 0; i < io_num.n_output; i++)
	{
		output_attrs[i].index = i;
		ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
		dump_tensor_attr(&(output_attrs[i]));
	}

	int channel = 3, width = 0, height = 0;
	if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
	{
		printf("model is NCHW input fmt\n");
		channel = input_attrs[0].dims[1];
		height = input_attrs[0].dims[2];
		width = input_attrs[0].dims[3];
	}
	else
	{
		printf("model is NHWC input fmt\n");
		height	= input_attrs[0].dims[1];
		width	= input_attrs[0].dims[2];
		channel = input_attrs[0].dims[3];
	}

	printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

	memset(inputs, 0, sizeof(inputs));
	inputs[0].index = 0;
	inputs[0].type = RKNN_TENSOR_UINT8;
	inputs[0].size = width * height * channel;
	inputs[0].fmt = RKNN_TENSOR_NHWC;
	inputs[0].pass_through = 0;
	printf("初始化rknn模型库完成.....\n");
	g_init=true;
	return 0;
}

RKnn info={0,0,0,0};

//识别一个图片,返回一个字符串用,分割,依次是left,top,width,heigh
int interRknn(char *input_path, char *output)
{	
	info={0,0,0,0};
	if(!g_init)
	{
		printf("请首先初始化识别库.......\n");
		return 0;
	}
	const float nms_threshold 	= NMS_THRESH;		// 默认的NMS阈值
	float box_conf_threshold	= BOX_THRESH;		// 默认的置信度阈值
	
	struct timeval start_time, stop_time, lastTime;
	//if(g_bExit)	{	break;}		//std::vector<std::string> v; sendAddr
	printf("开始识别图片数据.......\n");
	//strcpy(input_path, (char*)buff);//	strcpy(input_path, "t1197.jpg");

	for(int i=0; i<100; i++)
	{
		if (!fs::exists(input_path)) {
			printf("没有这个图片=%s ..................\n", input_path);
		}
		else
			break;
	}
	gettimeofday(&start_time, NULL);
	// 读取图片
	printf("读取图片 Read=%s ...\n", input_path);
	cv::Mat orig_img = cv::imread(input_path, 1);
	if (!orig_img.data)
	{
		printf("cv::imread %s fail! ..............\n", input_path);
		return 0;
	}

	cv::Mat img;
	cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
	int width 	= img.cols;
	int height 	= img.rows;
	printf("img width = %d, img height = %d\n", width, height);

	// 指定目标大小和预处理方式,默认使用 LetterBox 的预处理
	BOX_RECT pads;
	memset(&pads, 0, sizeof(BOX_RECT));
	cv::Size target_size(width, height);
	cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);

	// 计算缩放比例
	float scale_w = (float)target_size.width / img.cols;
	float scale_h = (float)target_size.height/ img.rows;

	inputs[0].buf = img.data;

	rknn_inputs_set(ctx, io_num.n_input, inputs);
	rknn_output outputs[io_num.n_output];
	memset(outputs, 0, sizeof(outputs));
	for (int i = 0; i < io_num.n_output; i++)
	{
		outputs[i].want_float = 0;
	}

	// 执行推理
	int ret = rknn_run(ctx, NULL);
	ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
	gettimeofday(&stop_time, NULL);
	printf("执行推理 once run use %0.1f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

	// 后处理
	detect_result_group_t detect_result_group;
	std::vector<float> out_scales;
	std::vector<int32_t> out_zps;
	for (int i = 0; i < io_num.n_output; ++i)
	{
		out_scales.push_back(output_attrs[i].scale);
		out_zps.push_back(output_attrs[i].zp);
	}

	post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
		box_conf_threshold, nms_threshold, pads, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
	
	if(detect_result_group.count==0)//如果没有结果就返回
	{			
		printf("no reselt 给跟踪程序发回了应答。。\n\n\n\n");
		return 0;
	}

	gettimeofday(&lastTime, NULL);
	printf("加上 后处理 once run use %0.1f ms\n", (__get_us(lastTime) - __get_us(start_time)) / 1000);

	// 画框和概率
	char text[256];
	char out_path[]={"out.jpg"};
	float fMaxProp=0;
	ret=0;

	for (int i = 0; i < detect_result_group.count; i++)
	{
		detect_result_t *p = &(detect_result_group.results[i]);
		printf("结果%s @ (%d %d %d %d) %f\n", p->name, p->box.left, p->box.top,p->box.right, p->box.bottom, p->prop);

		info.w  = p->box.right	 - p->box.left;
		info.h	= p->box.bottom - p->box.top;

		printf("结果%d  w=%d h=%d\n", i, info.w, info.h);
		if(info.w>10 && info.h>10)
		{
			if(p->prop> fMaxProp)
			{
				ret=i;
				fMaxProp= p->prop;
			}
		}
		/*sprintf(text, "%s %.1f%%", p->name, p->prop * 100);
		int x1 = p->box.left;
		int y1 = p->box.top;
		int x2 = p->box.right;
		int y2 = p->box.bottom;
		rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(256, 0, 0, 256), 3);
		putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));*/
	}		

	detect_result_t *pr=&(detect_result_group.results[ret]);
	sprintf(text, "%s %.1f%%", pr->name, pr->prop * 100);

	info.left=pr->box.left;
	info.top= pr->box.top;
	info.w  = pr->box.right	 - pr->box.left;
	info.h	= pr->box.bottom - pr->box.top;
	//strcpy(info.name,out_path);

	int x1 = pr->box.left;
	int y1 = pr->box.top;
	int x2 = pr->box.right;
	int y2 = pr->box.bottom;
	rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(256, 0, 0, 256), 3);
	putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));

	gettimeofday(&lastTime, NULL);
	printf("加上 画框 once run use %0.1f ms\n", (__get_us(lastTime) - __get_us(start_time)) / 1000);

	imwrite(out_path, orig_img);	//保存一个加边框的图片./out.jpg
	ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
	
	printf("识别了一次  %s rknn_outputs_release=%d  ........\n\n\n\n", out_path, ret);
	sprintf(output, "%d,%d,%d,%d", x1,y1, info.w, info.h);
	return 1;
}
}
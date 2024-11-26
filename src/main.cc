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

#include "RgaUtils.h"
#include "postprocess.h"
#include "rknn_api.h"
#include "preprocess.h"

//识别出的目标结构
typedef struct BOXX
{
    int left;		//目标左
	int top;		//目标上
    int right;		//目标右
    int bottom;		//目标下
	char name[4];	//目标类型
	float thresh;	//目标可信度值0-100
} ;

//识别结构结构
struct RKnn
{
	unsigned char num;	//目标个数，0表示没有目标，最多5个目标，每个目标占用20个字节
	BOXX box[OBJ_NUMB_MAX_SIZE];	
};

namespace fs = boost::filesystem;
using namespace std;

#define PERF_WITH_POST 1

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

char  g_HostQtIP[]			={"127.0.0.1"};
/*--   Main Functions---------------------------*/
extern "C"{ 
int main(int argc, char **argv)
{
	struct timeval start_time, stop_time, lastTime;
	//gettimeofday(&start_time, NULL);
	
	struct sockaddr_in addr, sendAddr, addr_from;   //本机地址

	//创建udp通信socket
    int g_udpQTfd = socket(AF_INET, SOCK_DGRAM, 0), on=1;;
    if(g_udpQTfd== -1)
    {
         perror("g_udpQTfd socket open failed!\n");
         return 0;
    }
    
	int nRet = setsockopt(g_udpQTfd, SOL_SOCKET, SO_BROADCAST, &on, sizeof(on));
	if(nRet< 0)
	{
		fprintf(stderr, "g_udpQTfd setsock opt failed ! \n");
		return 0;
	}

	memset(&addr, 0, sizeof(addr));
   	addr.sin_family = AF_INET;                 		//使用IPv4协议
    addr.sin_port = htons(5001);               		//设置接收端口号
    addr.sin_addr.s_addr = inet_addr(g_HostQtIP);	//设置接收IP
	
	nRet = bind(g_udpQTfd, (struct sockaddr *)&addr, sizeof(sockaddr_in));
	if(nRet < 0)
	{
		fprintf(stderr, "绑定端口 failed rknn程序已经启动了! nRet=%d\n", nRet);
		return 0;
	}

	memcpy(&sendAddr, &addr, sizeof(addr));
	sendAddr.sin_port = htons(5002);		//发送端口号

	int ret;
	rknn_context ctx;
	size_t actual_size = 0;
	int img_width = 0, img_height = 0, img_channel = 0;//图片宽度，高度，通道id
	const float nms_threshold 	= NMS_THRESH;		// 默认的NMS阈值
	float box_conf_thresh		= BOX_THRESH;		// 默认的置信度阈值
	
	char model_name[]={"best-3.rknn"};// (char *)argv[1];			//输入的model file
	//char model_name[]={"yolov5s.rknn"};// (char *)argv[1];			//输入的model file

	char input_path[128] ={0};			// argv[2];				//输入的识别图片参数
	std::string out_path	= "./out.jpg";     //识别后输出的文件

	box_conf_thresh=atof(argv[1]);	// 默认的置信度阈值
	// init rga context
	rga_buffer_t src;		
	rga_buffer_t dst;
	memset(&src, 0, sizeof(src));		memset(&dst, 0, sizeof(dst));

	printf("model name=%s box_conf_threshold = %.2f, nms_threshold = %.2f\n", model_name,box_conf_thresh, nms_threshold);

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

	rknn_input_output_num io_num;
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

	rknn_tensor_attr output_attrs[io_num.n_output];
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
		channel=input_attrs[0].dims[1];
		height =input_attrs[0].dims[2];
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

	rknn_input inputs[1];
	memset(inputs, 0, sizeof(inputs));
	inputs[0].index = 0;
	inputs[0].type = RKNN_TENSOR_UINT8;
	inputs[0].size = width * height * channel;
	inputs[0].fmt = RKNN_TENSOR_NHWC;
	inputs[0].pass_through = 0;
	
	socklen_t len= sizeof(addr_from);
	while(1)
	{
		//if(g_bExit)	{	break;}		//std::vector<std::string> v; sendAddr
		printf("循环等待接收识别图片数据...\n");
		uint8_t buff[512]={0};	
		int nread = recvfrom(g_udpQTfd, (char*)buff, sizeof(buff), 0, (sockaddr *)&addr_from, &len); 
		if(nread<5)
		{
			printf("recvfrom 数据量太少 nread=%d\n", nread);
			continue;
		}

		strcpy(input_path, (char*)buff);//	strcpy(input_path, "t1197.jpg");

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
			printf("cv::imread %s fail!\n", input_path);
			continue;
		}

		cv::Mat img;
		cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
		img_width 	= img.cols;
		img_height 	= img.rows;
		printf("img width = %d, img height = %d\n", img_width, img_height);

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
		ret = rknn_run(ctx, NULL);
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
			box_conf_thresh, nms_threshold, pads, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

		RKnn info={0};
		memset(&info, 0, sizeof(info));

		if(detect_result_group.count==0)//如果没有结果就返回
		{			
			sendto(g_udpQTfd, &info, sizeof(RKnn), 0, (struct sockaddr *)&sendAddr,sizeof(sockaddr_in));
			printf("no reselt 给跟踪程序发回了应答回令 。。。。。\n\n\n\n");
			continue;
		}

		gettimeofday(&lastTime, NULL);
		printf("加上 后处理 once run use %0.1f ms\n", (__get_us(lastTime) - __get_us(start_time)) / 1000);

		for(int i=0; i<detect_result_group.count; i++)
		{
			// 画框和概率
			//char text[256];			//float fMinProp= box_conf_thresh *10, thresh=0;
			//int x1=0,y1=0,x2=0,y2=0;
			//for (int i = 0; i < detect_result_group.count; i++)
			detect_result_t *p = &(detect_result_group.results[i]);
			printf("结果%s @ (%d %d %d %d) %.2f name=%s\n", p->name, p->box.left, p->box.top,p->box.right, p->box.bottom, p->prop, p->name);

			int w= p->box.right	- p->box.left;
			int h= p->box.bottom- p->box.top;
			
			if(p->box.left>10 && p->box.top>10 && w>10 && w<380 && h>10 && h<400 && p->prop > box_conf_thresh)
			{
				printf("结果  w=%d h=%d name=%s\n", w, h, p->name);
				info.box[info.num].left  =p->box.left;
				info.box[info.num].top   =p->box.top;
				info.box[info.num].right =p->box.right;
				info.box[info.num].bottom=p->box.bottom;
				info.box[info.num].thresh=p->prop;			
				strcpy(info.box[info.num].name, p->name);
				info.num =info.num+1;
			}							
		}
		
		if(info.num==0)
		{			
			sendto(g_udpQTfd, &info, sizeof(RKnn), 0, (struct sockaddr *)&sendAddr,sizeof(sockaddr_in));
			printf("no reselt 给跟踪程序发回了应答回令 。。。。。\n\n\n\n");
			continue;
		}

		//rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(256, 0, 0, 256), 3);
		//putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
		/*	info.left=x1;
		info.top= y1;
		info.w  = x2 -x1;
		info.h	= y2 -y1;
		info.thresh= p->prop *10;
		strcpy(info.name,out_path.c_str());*/
		//gettimeofday(&lastTime, NULL);
		//printf("加上 画框 once run use %0.1f ms\n", (__get_us(lastTime) - __get_us(start_time)) / 1000);
		//imwrite(out_path, orig_img);	//保存一个加边框的图片./out.jpg
		ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
		ret =sendto(g_udpQTfd, &info, sizeof(RKnn), 0, (struct sockaddr *)&sendAddr,sizeof(sockaddr_in));
		printf("给跟踪程序发回了应答  字节个数=%d........\n\n\n\n", ret);
	}
	printf("跟踪程序异常退出了....\n");
	// 耗时统计
	deinitPostProcess();
		// release
	ret = rknn_destroy(ctx);
	if (model_data)	free(model_data);
	return 0;
}
}

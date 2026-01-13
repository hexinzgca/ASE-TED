#!/bin/sh

# 1. 定义变量
jobflag=$1  			# 任务标识，读取第一个参数
jobnum=20  				# 任务总数，可根据实际需求修改
ncore=4  				# 指定并发进程数，限制同时运行的任务数
export OMP_THREAD_NUM=4 # ncore*OMP_THERAD_NUM不超过机器的总核数 

# 2. 判断文件夹 & 数据状态
if [ "${jobflag}" != "small1" -a "${jobflag}" != "system1" ]; then
	echo "任务标识 ${jobflag} 不支持"
	exit -1
fi

if [ ! -e "reaxff_${jobflag}" ]; then
	echo "任务标识数据缺失 ${jobflag}"
	exit -1
fi

# 3. 初始化并发进程计数器
process_count=0

# 4. 循环执行任务，控制并发数
for ((i=1; i<=$jobnum; i++)); do
    idir="reaxff_${jobflag}_${i}"
    
    # 前置：创建目录并拷贝文件（原有逻辑保留）
    if [ ! -e "$idir" ]; then
        mkdir -p "$idir"
        cp reaxff_${jobflag}/oplsaa2_react.data "$idir"
        cp reaxff_${jobflag}/neff.neff "$idir"
        cp reaxff_${jobflag}/reaxff.ff "$idir"
    fi

    echo "启动任务 id=${i}，运行目录=${idir}，当前并发进程数=${process_count}"
    
    # 核心：nohup后台运行python脚本，重定向日志，&放入后台
    nohup python ../scripts/run_reaxff.py -f $idir > "${idir}/task_${i}.log" 2>&1 &
    # nohup sleep 1 && echo "xxxx" 2>&1 & # for test
    
    # 5. 并发控制：计数器+1，判断是否达到最大并发数
    process_count=$((process_count + 1))
    if [ ${process_count} -eq ${ncore} ]; then
        # 等待所有后台进程完成（阻塞当前循环，直到所有后台进程结束）
        wait
        # 重置计数器，继续下一批并发任务
        process_count=0
        echo "一批任务执行完成，开始下一批..."
    fi
done

# 6. 最后一批任务：等待剩余未完成的后台进程（避免遗漏任务）
wait
echo "所有任务全部执行完成！"




package demo.IEC104;

import demo.IEC104.content.*;
import lombok.AllArgsConstructor;

import java.util.Map;

/**
 * @author bin
 * @version 1.0.0
 * @see <a href="https://blog.csdn.net/changqing1990/article/details/134327980">详解IEC104 规约【最详细版】</a>
 * @since 2024/09/11
 */
@AllArgsConstructor
public enum ContentLayout {
    NULL(0),
    /**
     * 在控制方向上，该地址为目标地址，监控方向上，该地址为源地址。
     * <p>
     * 通常情况下，IOA 的地址范围被限制到最大35535（2 byte）。在特殊情况下，IOA的第三个字节仅用于结构化信息对象地址的情况，以便在特定系统中定义明确的地址。
     * 如果ASDU 的信息对象地址，没有被用，将被设置为
     * 0。
     */
    IOA(3) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new IOA(content, offset);
        }
    },
    // region 1监视方向的过程信息
    /**
     * 有质量描述符的单点信息
     */
    SIQ(1) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new SIQ(content[offset]);
        }
    },
    /**
     * 有质量描述符的双点信息
     */
    DIQ(1) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new DIQ(content[offset]);
        }
    },
    /**
     * 二进制状态信息
     */
    BSI(4) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new BSI(content, offset);
        }
    },
    /**
     * 状态更改探测
     */
    SCD(4) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new SCD(content, offset);
        }
    },
    /**
     * 质量描述符
     */
    QDS(1) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new QDS(content[offset]);
        }
    },
    /**
     * 具有瞬态指示值
     */
    VTI(1) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new VTI(content[offset]);
        }
    },
    /**
     * 归一化值
     */
    NVA(2) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new NVA(content, offset);
        }
    },
    /**
     * 标度值
     */
    SVA(2) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new SVA(content, offset);
        }
    },
    /**
     * 短浮点数
     */
    IEEE_STD_754(4) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new IEEE_STD_754(content, offset);
        }
    },
    /**
     * 二进制计数器读数
     */
    BCR(5) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new BCR(content, offset);
        }
    },
    // endregion
    // region 2保护
    /**
     * 保护设备的单个事件
     */
    SEP(1) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new SEP(content[offset]);
        }
    },
    /**
     * 保护设备启动事件
     */
    SPE(1),
    /**
     * 保护设备输出电路信息
     */
    OCI(1) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new OCI(content[offset]);
        }
    },
    /**
     * 保护设备事件的质量描述符
     */
    QDP(1) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new QDP(content[offset]);
        }
    },
    // endregion
    // region 3命令
    /**
     * 单个命令
     */
    SCO(1) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new Command(content[offset]);
        }
    },
    /**
     * 双命令
     */
    DCO(1) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new Command(content[offset]);
        }
    },
    /**
     * 调节阶跃指令
     */
    RCO(1) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new Command(content[offset]);
        }
    },
    // endregion
    // region 4时间
    /**
     * 七字节二进制时间
     */
    CP56Time2a(7) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new CP56Time2a(content, offset);
        }
    },
    /**
     * 三字节二进制时间
     */
    CP32Time2a(4) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new CP32Time2a(content, offset);
        }
    },
    /**
     * 三字节二进制时间
     */
    CP24Time2a(3) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new CP24Time2a(content, offset);
        }
    },
    /**
     * 二字节二进制时间
     */
    CP16Time2a(2) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new CP16Time2a(content, offset);
        }
    },
    // endregion
    // region 5限定符
    /**
     * 讯问资格
     */
    QOI(1) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new QOI(content[offset]);
        }
    },
    /**
     * 反询问命令限定符
     */
    QCC(1),
    /**
     * 测量值参数限定符
     */
    QPM(1) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new QPM(content[offset]);
        }
    },
    /**
     * 参数激活的限定符
     */
    QPA(1) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new QPA(content[offset]);
        }
    },
    /**
     * 重置过程命令的限定符
     */
    QRP(1),
    /**
     * 命令的限定符
     */
    QOC(1) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new Command(content[offset]);
        }
    },
    /**
     * 设定值命令的限定符
     */
    QOS(1) {
        @Override
        public BaseContent parseContent(byte[] content, int offset) {
            return new QOS(content[offset]);
        }
    },
    // endregion
    // region 6文件传输
    /**
     * 文件就绪限定符
     */
    FRQ(1),
    /**
     * 节就绪限定符
     */
    SRQ(1),
    /**
     * 选择并调用限定符
     */
    SCQ(1),
    /**
     * 最后一节或段限定符
     */
    LSQ(1),
    /**
     * 确认文件或节限定符
     */
    AFQ(1),
    /**
     * 文件名
     */
    NOF(2),
    /**
     * 段命名
     */
    NOS(2),
    /**
     * 文件/段的长度
     */
    LOF(3),
    /**
     * 分段长度
     */
    LOS(1),
    /**
     * Checksum
     */
    CHS(1),
    /**
     * 文件状态
     */
    SOF(1),
    // endregion
    // region 7杂项
    /**
     * 初始化原因
     */
    COI(1),
    /**
     * 固定测试位模式，两个八位字节
     */
    FBP(1),
    // endregion
    ;
    public final int length;

    public BaseContent parseContent(byte[] content, int offset) {
        return new Unknown(content, offset, length, name());
    }

    public void parseContent(byte[] content, int offset, Map<String, Object> map) {
        map.put(name(), parseContent(content, offset));
    }
}

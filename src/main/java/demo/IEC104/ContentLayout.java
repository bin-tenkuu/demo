package demo.IEC104;

import lombok.AllArgsConstructor;
import lombok.val;

import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.Map;

/**
 * @author bin
 * @version 1.0.0
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
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            map.put("地址(IOA)", ByteUtil.getShort(content, offset));
        }
    },
    // region 1监视方向的过程信息
    /**
     * 有质量描述符的单点信息
     */
    SIQ(1) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val b = content[offset];
            parseByte(b, map);
            map.put("遥信状态值(SPI)", ByteUtil.getBit(b, 0));
        }
    },
    /**
     * 有质量描述符的双点信息
     */
    DIQ(1) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val b = content[offset];
            parseByte(b, map);
            map.put("遥信状态值(DPI)", b & 3);
        }
    },
    /**
     * 二进制状态信息
     */
    BSI(4) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            map.put("数据(BSI)", ByteUtil.getInt(content, offset));
        }
    },
    /**
     * 状态更改探测
     */
    SCD(4) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            map.put("数据(SCD)", ByteUtil.getInt(content, offset));
        }
    },
    /**
     * 质量描述符
     */
    QDS(1) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val b = content[offset];
            parseByte(b, map);
            map.put("溢出标志位(OV)", ByteUtil.getBit(b, 0));
        }
    },
    /**
     * 具有瞬态指示值
     */
    VTI(1) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val b = content[offset];
            map.put("瞬变状态", ByteUtil.getBit(b, 7));
            map.put("数据(VTI)", b << 1 >> 1);
        }
    },
    /**
     * 归一化值
     */
    NVA(2) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val s = ByteUtil.getShort(content, offset);
            map.put("SVA", s);
            map.put("归一值(NVA)", s / 32768F);
        }
    },
    /**
     * 标度值
     */
    SVA(2) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            map.put("标量值(SVA)", ByteUtil.getShort(content, offset));
        }
    },
    /**
     * 短浮点数
     */
    IEEE_STD_754(4) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            map.put("数据(IEEE_STD_754)", ByteUtil.getFloat(content, offset));
        }
    },
    /**
     * 二进制计数器读数
     */
    BCR(5) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val b = content[offset];
            map.put("计数量被调整(CA)", ByteUtil.getBit(b, 6));
            map.put("是否有进位(CY)", ByteUtil.getBit(b, 5));
            map.put("顺序号(SQ)", b & 0x1f);
            map.put("读数(BCR)", ByteUtil.getInt(content, offset + 1));
        }
    },
    // endregion
    // region 2保护
    /**
     * 保护设备的单个事件
     */
    SEP(1) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val b = content[offset];
            map.put("反向保护启动(SRD)", ByteUtil.getBit(b, 5));
            map.put("接地电流保护启动(SIE)", ByteUtil.getBit(b, 4));
            map.put("C相保护启动(SL3)", ByteUtil.getBit(b, 3));
            map.put("B相保护启动(SL2)", ByteUtil.getBit(b, 2));
            map.put("A相保护启动(SL1)", ByteUtil.getBit(b, 1));
            map.put("总启动(GS)", ByteUtil.getBit(b, 0));
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
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val b = content[offset];
            map.put("C相保护命令输出至输出电路(SL3)", ByteUtil.getBit(b, 3));
            map.put("B相保护命令输出至输出电路(SL2)", ByteUtil.getBit(b, 2));
            map.put("A相保护命令输出至输出电路(SL1)", ByteUtil.getBit(b, 1));
            map.put("总命令输出至输出电路(GC)", ByteUtil.getBit(b, 0));
        }
    },
    /**
     * 保护设备事件的质量描述符
     */
    QDP(1) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val b = content[offset];
            parseByte(b, map);
            map.put("时间间隔值无效(EI)", ByteUtil.getBit(b, 3));
            map.put("事件", b & 3);
        }
    },
    // endregion
    // region 3命令
    /**
     * 单个命令
     */
    SCO(1) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val b = content[offset];
            parseCO(b, map);
        }
    },
    /**
     * 双命令
     */
    DCO(1) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val b = content[offset];
            parseCO(b, map);
        }
    },
    /**
     * 调节阶跃指令
     */
    RCO(1) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val b = content[offset];
            parseCO(b, map);
        }
    },
    // endregion
    // region 4时间
    /**
     * 七字节二进制时间
     */
    CP56Time2a(7) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val second = ByteUtil.getShort(content, offset);
            val b2 = content[offset + 2];
            val b4 = content[offset + 4];
            val b3 = content[offset + 3];
            val time = LocalDateTime.of(
                    content[offset + 6] & 0b01111111,
                    content[offset + 5] & 0b00001111,
                    b4 & 0b00011111,
                    b3 & 0b00011111,
                    b2 & 0b00111111,
                    second / 1000,
                    (second % 1000) * 1000000
            );
            map.put("是否有效(IV)", ByteUtil.getBit(b2, 7));
            map.put("夏令时(SU)", ByteUtil.getBit(b3, 7));
            map.put("时间(CP56Time2a)", time);
            map.put("星期(CP56Time2a)", b4 >>> 5);
        }
    },
    /**
     * 三字节二进制时间
     */
    CP24Time2a(3) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val second = ByteUtil.getShort(content, offset);
            val b2 = content[offset + 2];
            val time = LocalTime.of(
                    0,
                    b2 & 0b00111111,
                    second / 1000,
                    (second % 1000) * 1000000
            );
            map.put("是否有效(IV)", ByteUtil.getBit(b2, 7));
            map.put("时间(CP24Time2a)", time);
        }
    },
    /**
     * 二字节二进制时间
     */
    CP16Time2a(2) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            map.put("时间(CP16Time2a)", ByteUtil.getShort(content, offset));
        }
    },
    // endregion
    // region 5限定符
    /**
     * 讯问资格
     */
    QOI(1) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            map.put("QOI", 0x14);
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
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val b = content[offset];
            map.put("是否运行", ByteUtil.getBit(b, 7));
            map.put("是否被改变", ByteUtil.getBit(b, 6));
            map.put("类型", b & 0x3f);
        }
    },
    /**
     * 参数激活的限定符
     */
    QPA(1) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            map.put("激活/停止(QPA)", content[offset]);
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
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val b = content[offset];
            parseCO(b, map);
        }
    },
    /**
     * 设定值命令的限定符
     */
    QOS(1) {
        @Override
        public void parseContent(byte[] content, int offset, Map<String, Object> map) {
            val b = content[offset];
            map.put("S/E", ByteUtil.getBit(b, 7));
            map.put("QOS", b << 1 >>> 1);
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

    private static void parseByte(byte b, Map<String, Object> map) {
        map.put("是否有效(IV)", ByteUtil.getBit(b, 7));
        map.put("刷新标志位(NT)", ByteUtil.getBit(b, 6));
        map.put("取代标志位(SB)", ByteUtil.getBit(b, 5));
        map.put("封锁标志位(BL)", ByteUtil.getBit(b, 4));
    }

    private static void parseCO(byte b, Map<String, Object> map) {
        map.put("选择/执行(S/E)", ByteUtil.getBit(b, 7));
        map.put("CO", b << 1 >>> 3);
        map.put("CS", b & 3);
    }

    public void parseContent(byte[] content, int offset, Map<String, Object> map) {
        if (length == 0) {
            return;
        }
        val bs = new byte[length];
        System.arraycopy(content, offset, bs, 0, length);
        map.put(name(), ByteUtil.toString(bs));
    }
}

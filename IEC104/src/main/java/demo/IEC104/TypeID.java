package demo.IEC104;

import lombok.val;

import java.util.Objects;

import static demo.IEC104.ContentLayout.*;

public enum TypeID {
    UNKOWN(0, "未定义"),
    // region 1监视方向上的过程信息
    M_SP_NA_1(1, "单点遥信", new ContentLayout[]{
            SIQ
    }),
    M_SP_TA_1(2, "带短时标的单点遥信", new ContentLayout[]{
            SIQ
    }, CP24Time2a),
    M_DP_NA_1(3, "双点遥信", new ContentLayout[]{
            DIQ
    }),
    M_DP_TA_1(4, "带短时标的双点遥信", new ContentLayout[]{
            DIQ
    }, CP24Time2a),
    M_ST_NA_1(5, "步长位置信息", new ContentLayout[]{
            VTI, QDS
    }),
    M_ST_TA_1(6, "带短时标的步长位置信息", new ContentLayout[]{
            VTI, QDS
    }, CP24Time2a),
    M_BO_NA_1(7, "32比特串", new ContentLayout[]{
            BSI, QDS
    }),
    M_BO_TA_1(8, "带短时标的32比特串", new ContentLayout[]{
            BSI, QDS
    }, CP24Time2a),
    M_ME_NA_1(9, "归一化值", new ContentLayout[]{
            NVA, QDS
    }),
    M_ME_TA_1(10, "带短时标的归一化值", new ContentLayout[]{
            NVA, QDS
    }, CP24Time2a),
    M_ME_NB_1(11, "标度化值", new ContentLayout[]{
            SVA, QDS
    }),
    M_ME_TB_1(12, "带短时标的标度化值", new ContentLayout[]{
            SVA, QDS
    }, CP24Time2a),
    M_ME_NC_1(13, "短浮点数", new ContentLayout[]{
            IEEE_STD_754, QDS
    }),
    M_ME_TC_1(14, "带短时标的短浮点数", new ContentLayout[]{
            IEEE_STD_754, QDS
    }, CP24Time2a),
    M_IT_NA_1(15, "累计值", new ContentLayout[]{
            BCR
    }),
    M_IT_TA_1(16, "带短时标的累计值", new ContentLayout[]{
            BCR
    }, CP24Time2a),
    M_EP_TA_1(17, "带时标的保护设备事件", new ContentLayout[]{
            QDP
    }, CP16Time2a),
    M_EP_TB_1(18, "带时标的继电保护装置成组启动事件", new ContentLayout[]{
            SEP, QDP
    }, CP16Time2a),
    M_EP_TC_1(19, "带时标的继电保护装置成组输出电路信息", new ContentLayout[]{
            OCI
    }, CP16Time2a),
    M_PS_NA_1(20, "具有状态变位检出的成组单点遥信", new ContentLayout[]{
            SCD, QDS
    }),
    M_ME_ND_1(21, "不带品质描述的归一化值", new ContentLayout[]{
            NVA
    }),
    M_SP_TB_1(30, "带CP56Time2a时标的单点遥信", new ContentLayout[]{
            SIQ
    }, CP56Time2a),
    M_DP_TB_1(31, "带CP56Time2a时标的双点遥信", new ContentLayout[]{
            DIQ
    }, CP56Time2a),
    M_ST_TB_1(32, "带CP56Time2a时标的步位置信息", new ContentLayout[]{
            VTI, QDS
    }, CP56Time2a),
    M_BO_TB_1(33, "带CP56Time2a时标的32比特串", new ContentLayout[]{
            BSI, QDS
    }, CP56Time2a),
    M_ME_TD_1(34, "带CP56Time2a时标的归一化值", new ContentLayout[]{
            NVA, QDS
    }, CP56Time2a),
    M_ME_TE_1(35, "带CP56Time2a时标的标度化值", new ContentLayout[]{
            SVA, QDS
    }, CP56Time2a),
    M_ME_TF_1(36, "带CP56Time2a时标的短浮点数", new ContentLayout[]{
            IEEE_STD_754, QDS
    }, CP56Time2a),
    M_IT_TB_1(37, "带CP56Time2a时标的累计量", new ContentLayout[]{
            BCR
    }, CP56Time2a),
    M_EP_TD_1(38, "带CP56Time2a时标的继电保护装置", new ContentLayout[]{
            QDP, CP16Time2a
    }, CP56Time2a),
    M_EP_TE_1(39, "带CP56Time2a时标的继电保护装置成组启动事件", new ContentLayout[]{
            SEP, QDP, CP16Time2a
    }, CP56Time2a),
    M_EP_TF_1(40, "带CP56Time2a时标的继电保护装置成组输出电炉信息", new ContentLayout[]{
            OCI, QDP, CP16Time2a
    }, CP56Time2a),
    /* 41~44 为将来的兼容定义保留*/
    // endregion
    // region 2控制方向上的过程信息
    C_SC_NA_1(45, "单点遥控", new ContentLayout[]{
            SCO
    }),
    C_DC_NA_1(46, "双点遥控", new ContentLayout[]{
            DCO
    }),
    C_RC_NA_1(47, "档位调节命令", new ContentLayout[]{
            RCO
    }),
    C_SE_NA_1(48, "设点命令-归一化值", new ContentLayout[]{
            NVA, QOS
    }),
    C_SE_NB_1(49, "设点命令-标度化值", new ContentLayout[]{
            SVA, QOS
    }),
    C_SE_NC_1(50, "设点命令-浮点数值", new ContentLayout[]{
            IEEE_STD_754, QOS
    }),
    C_BO_NA_1(51, "32比特串", new ContentLayout[]{
            BSI
    }),
    /* 52~57 为将来的兼容定义保留*/
    C_SC_TA_1(58, "带CP56Time2a时标的单点遥控", new ContentLayout[]{
            SCO, CP56Time2a
    }),
    C_DC_TA_1(59, "带CP56Time2a时标的双点遥控", new ContentLayout[]{
            DCO, CP56Time2a
    }),
    C_RC_TA_1(60, "带CP56Time2a时标的档位调节命令", new ContentLayout[]{
            RCO, CP56Time2a
    }),
    C_SE_TA_1(61, "带CP56Time2a时标的设点命令-归一化值", new ContentLayout[]{
            NVA, QOS, CP56Time2a
    }),
    C_SE_TB_1(62, "带CP56Time2a时标的设点命令-标度化值", new ContentLayout[]{
            SVA, QOS, CP56Time2a
    }),
    C_SE_TC_1(63, "带CP56Time2a时标的设点命令-浮点数值", new ContentLayout[]{
            IEEE_STD_754, QOS, CP56Time2a
    }),
    C_BO_TA_1(64, "带CP56Time2a时标的32比特串", new ContentLayout[]{
            BSI, CP56Time2a
    }),
    /* 65~69 为将来的兼容定义保留*/
    // endregion
    // region 3监视方向上的系统信息
    M_EI_NA_1(70, "初始化结束", new ContentLayout[]{
            COI
    }),
    /* 71~99 为将来的兼容定义保留*/
    // endregion
    // region 4控制方向上的系统信息
    C_IC_NA_1(100, "总召唤", new ContentLayout[]{
            QOI
    }),
    C_CI_NA_1(101, "电度总召唤", new ContentLayout[]{
            QCC
    }),
    C_RD_NA_1(102, "读命令"),
    C_CS_NA_1(103, "时钟同步命令", new ContentLayout[]{
            CP56Time2a
    }),
    C_TS_NA_1(104, "测试命令", new ContentLayout[]{
            FBP
    }),
    C_RP_NA_1(105, "复位进程命令", new ContentLayout[]{
            QRP
    }),
    C_CD_NA_1(106, "延时获得命令", new ContentLayout[]{

    }),
    C_TS_TA_1(107, "带CP56Time2a时标的测试命令", new ContentLayout[]{
            FBP, CP56Time2a
    }),
    /* 108~109 为将来的兼容定义保留*/
    // endregion
    // region 5控制方向上的参数
    P_ME_NA_1(110, "测量值参数,归一化值", new ContentLayout[]{
            NVA, QPM
    }),
    P_ME_NB_1(111, "测量值参数,标度化值", new ContentLayout[]{
            SVA, QPM
    }),
    P_ME_NC_1(112, "测量值参数,短浮点数", new ContentLayout[]{
            IEEE_STD_754, QPM
    }),
    P_AC_NA_1(113, "参数激活", new ContentLayout[]{
            QPA
    }),
    /* 114~119 为将来的兼容定义保留*/
    // endregion
    // region 6文件传输
    F_FR_NA_1(120, "文件准备就绪", new ContentLayout[]{
            NOF, LOF, FRQ
    }),
    F_SR_NA_1(121, "节准备就绪", new ContentLayout[]{
            NOF, NOS, LOF, SRQ
    }),
    F_SC_NA_1(122, "召唤目录、文件、节", new ContentLayout[]{
            NOF, NOS, SCQ
    }),
    F_LS_NA_1(123, "最后节、段", new ContentLayout[]{
            NOF, NOS, LSQ, CHS
    }),
    F_AF_NA_1(124, "确认文件、节", new ContentLayout[]{
            NOF, NOS, AFQ
    }),
    F_SG_NA_1(125, "段", new ContentLayout[]{
            NOF, NOS, LOS
    }),
    F_DR_TA_1(126, "目录", new ContentLayout[]{
            NOF, LOF, SOF, CP56Time2a //, segement
    }),
    F_SC_NB_1(127, "日志查询，请求存档文件"),
    // endregion
    /* 128~135 保留用于消息路由*/
    /* 136~255 用于特殊用途*/;
    private static final TypeID[] values = new TypeID[128];

    static {
        for (TypeID value : values()) {
            values[value.type] = value;
        }
    }

    public final byte type;
    public final String name;
    public final ContentLayout[] layout;
    public final ContentLayout timeLayout;

    TypeID(int type, String name) {
        this.type = (byte) type;
        this.name = name;
        this.layout = new ContentLayout[0];
        this.timeLayout = NULL;
    }

    TypeID(int type, String name, ContentLayout[] layout) {
        this.type = (byte) type;
        this.name = name;
        this.layout = layout;
        this.timeLayout = NULL;
    }

    TypeID(int type, String name, ContentLayout[] layout, ContentLayout timeLayout) {
        this.type = (byte) type;
        this.name = name;
        this.layout = layout;
        this.timeLayout = timeLayout;
    }

    public static TypeID getByType(byte type) {
        val value = values[type & 0xFF];
        return Objects.requireNonNullElse(value, UNKOWN);
    }

}

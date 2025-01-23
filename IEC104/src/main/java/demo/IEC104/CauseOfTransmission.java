package demo.IEC104;


import lombok.val;

import java.util.Objects;

public enum CauseOfTransmission {
    UNKOWN(0, "未定义"),
    CS101_COT_PERIODIC(1, "周期、循环"),
    CS101_COT_BACKGROUND_SCAN(2, "背景扫描"),
    CS101_COT_SPONTANEOUS(3, "突发(自发)"),
    CS101_COT_INITIALIZED(4, "初始化"),
    CS101_COT_REQUEST(5, "请求或被请求"),
    CS101_COT_ACTIVATION(6, "激活"),
    CS101_COT_ACTIVATION_CON(7, "激活确认"),
    CS101_COT_DEACTIVATION(8, "停止激活"),
    CS101_COT_DEACTIVATION_CON(9, "停止激活确认"),
    CS101_COT_ACTIVATION_TERMINATION(10, "激活终止"),
    CS101_COT_RETURN_INFO_REMOTE(11, "远方命令引起的返送信息"),
    CS101_COT_RETURN_INFO_LOCAL(12, "当地命令引起的返送信息"),
    CS101_COT_FILE_TRANSFER(13, "文件传输"),
    CS101_COT_AUTHENTICATION(14, "身份认证"),
    CS101_COT_MAINTENANCE_OF_AUTH_SESSION_KEY(15),
    CS101_COT_MAINTENANCE_OF_USER_ROLE_AND_UPDATE_KEY(16),
    /* 14-19 为配套标准兼容范围保留 */
    CS101_COT_INTERROGATED_BY_STATION(20, "响应第1组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_1(21, "响应第2组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_2(22, "响应第3组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_3(23, "响应第4组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_4(24, "响应第5组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_5(25, "响应第6组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_6(26, "响应第7组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_7(27, "响应第8组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_8(28, "响应第9组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_9(29, "响应第10组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_10(30, "响应第10组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_11(31, "响应第11组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_12(32, "响应第12组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_13(33, "响应第13组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_14(34, "响应第14组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_15(35, "响应第15组召唤"),
    CS101_COT_INTERROGATED_BY_GROUP_16(36, "响应第16组召唤"),
    CS101_COT_REQUESTED_BY_GENERAL_COUNTER(37, "响应计数量站召唤"),
    CS101_COT_REQUESTED_BY_GROUP_1_COUNTER(38, "响应第1组计数量召唤"),
    CS101_COT_REQUESTED_BY_GROUP_2_COUNTER(39, "响应第2组计数量召唤"),
    CS101_COT_REQUESTED_BY_GROUP_3_COUNTER(40, "响应第3组计数量召唤"),
    CS101_COT_REQUESTED_BY_GROUP_4_COUNTER(41, "响应第4组计数量召唤"),
    /* 42-43 为配套标准兼容范围保留 */
    CS101_COT_UNKNOWN_TYPE_ID(44, "未知类型标识"),
    CS101_COT_UNKNOWN_COT(45, "未知的传送原因"),
    CS101_COT_UNKNOWN_CA(46, "未知的 ASDU 公共地址"),
    CS101_COT_UNKNOWN_IOA(47, "未知的信息对象地址")
    /* 48-63 用于特殊用途（私有范围）*/;
    public final byte type;
    public final String name;
    private static final CauseOfTransmission[] values = new CauseOfTransmission[48];

    static {
        for (CauseOfTransmission value : values()) {
            values[value.type] = value;
        }
    }

    CauseOfTransmission(int type) {
        this.type = (byte) type;
        this.name = "";
    }

    CauseOfTransmission(int type, String name) {
        this.type = (byte) type;
        this.name = name;
    }

    public static CauseOfTransmission getByType(byte type) {
        val value = values[type & 0xFF];
        return Objects.requireNonNullElse(value, UNKOWN);
    }

}

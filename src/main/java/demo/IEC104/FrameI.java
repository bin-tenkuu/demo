package demo.IEC104;

import demo.IEC104.content.BaseContent;
import lombok.Getter;
import lombok.Setter;
import lombok.val;

import java.util.List;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/10
 */
@Getter
@Setter
public class FrameI extends Frame {
    private byte[] content = new byte[0];

    public FrameI(FrameType type, byte[] data) {
        super(type, data);
        if (data.length <= 12) {
            return;
        }
        val bytes = new byte[data.length - 12];
        System.arraycopy(data, 12, bytes, 0, bytes.length);
        content = bytes;
    }

    public FrameI() {
        super(FrameType.I, new byte[12]);
    }

    // region APCI
    public int getSendCounte() {
        return ByteUtil.getShort(data, 2) >> 1;
    }

    public void setSendCounte(int sendCounter) {
        ByteUtil.setShort(data, 2, (short) (sendCounter << 1));
    }

    public int getReceiveCounte() {
        return ByteUtil.getShort(data, 4) >> 1;
    }

    public void setReceiveCounte(int receiveCounter) {
        ByteUtil.setShort(data, 4, (short) (receiveCounter << 1));
    }

    // endregion

    /**
     * @return 对于sq == 1，会将IOA单独放入第一个list，时间戳放入最后一个list（如果有）
     */
    public List<List<BaseContent>> getContentsList() {
        return FrameUtil.parseContentsList(this);
    }

    public void setContentsList(List<List<BaseContent>> contentsList) {
        content = FrameUtil.toByteArray(contentsList, getSq(), TypeID.getByType(getTypeId()).timeLayout);
    }

    @Override
    public byte[] toByteArray() {
        val bytes = new byte[12 + content.length];
        System.arraycopy(data, 0, bytes, 0, 12);
        System.arraycopy(content, 0, bytes, 12, content.length);
        return bytes;
    }

    // region ASDU

    /**
     * 重要的是要注意，类型标识适用于整个 ASDU，因此如果 ASDU 中包含多个信息对象，则它们都属于同一类型。
     *
     * @return 类型标识 (TypeID，1 字节)
     */
    public byte getTypeId() {
        return data[6];
    }

    public void setTypeId(byte typeId) {
        data[6] = typeId;
    }

    /**
     * SQ=0（信息对象序列）：对多个同一类型的信息对象（IO）中的单个信息元素或信息元素组合进行寻址。
     * <p>
     * 每个单个元素或元素组合都由信息对象地址寻址。ASDU 可能由一个或多个相等的信息对象组成。对象数采用二进制编码（对象数），并定义信息对象的数量。
     * SQ=0 表示信息对象序列，其中每个对象都有自己的信息对象地址。信息对象的数量由Number Of Objects七位值给出。因此，此 ASDU 中最多可以有 127 个信息对象。
     * <p>
     * SQ=1（仅一个信息对象）：按 ASDU 对单个信息元素序列或单个对象的信息元素相等组合进行寻址，见下图。
     * <p>
     * 信息对象地址寻址相等信息对象序列（例如，相同格式的测量值）。信息对象地址指定序列中第一个信息元素的关联地址。后续信息元素通过从此偏移量开始连续 +1 的数字来标识。对象数量是二进制编码的（元素数量），并定义信息元素的数量。对于信息元素序列，每个 ASDU 仅分配一个信息对象。
     * 当 SQ=1 时，该结构在一个信息对象内包含一系列信息元素。所有信息对象都具有相同的格式，例如测量值。只有一个信息对象地址，即第一个信息元素的地址。
     *
     * @return SQ（结构限定符）位指定如何寻址信息对象或元素。
     */
    public boolean getSq() {
        return ByteUtil.getBit(data[7], 7);
    }

    public void setSq(boolean sq) {
        data[7] = ByteUtil.setBit(data[7], 7, sq);
    }

    /**
     * 使用范围 0 – 127
     * 0 表示 ASDU 不包含信息对象 (IO)
     * 1-127 定义信息对象或元素的数量
     *
     * @return 对象/元素的数量
     */
    public int getNumber() {
        return data[7] & 0x7F;
    }

    public void setNumber(int number) {
        data[7] = (byte) (data[7] & 0x80 | number);
    }

    /**
     * T（测试）位定义了在测试条件下生成的 ASDU，其目的不是控制过程或改变系统状态。
     *
     * @return 测试标记
     */
    public boolean getT() {
        return ByteUtil.getBit(data[8], 7);
    }

    public void setT(boolean t) {
        data[8] = ByteUtil.setBit(data[8], 7, t);
    }

    /**
     * P/N（积极/消极）位指示主要应用功能请求的激活的肯定或否定确认。
     * P/N = 0（积极确认），P/N = 1（消极确认）。
     * P/N 与控制命令一起使用时有意义。当控制命令在监控方向上镜像时，将使用该位，它指示命令是否已执行。当 PN 位不相关时，将其设置为零。
     *
     * @return 积极/消极确认
     */
    public boolean getPn() {
        return ByteUtil.getBit(data[8], 6);
    }

    public void setPn(boolean pn) {
        data[8] = ByteUtil.setBit(data[8], 6, pn);
    }

    /**
     * COT 字段用于控制通信网络上和站内消息的路由，通过 ASDU 指向正确的程序或任务进行处理。控制方向的 ASDU 是确认的应用服务，可以在监控方向镜像，传输原因不同。
     * COT 是一个六位代码，用于解释目标站的信息。每个定义的 ASDU 类型都有一个定义的代码子集，这些代码对其有意义。
     * 0 未定义，1-47 用于此配套标准的标准定义（兼容范围），48-63 用于特殊用途（私有范围）。
     *
     * @return 传输原因（COT）
     */
    public byte getCot() {
        return (byte) (data[8] & 0b0011_1111);
    }

    public void setCot(byte cot) {
        data[8] = (byte) (data[8] & 0b1100_0000 | cot);
    }

    /**
     * 源发站地址在系统基础上是可选的。它为控制站提供了一种明确标识自身的方式。当系统中只有一个控制站时，这不是必需的，但当有多个控制站或某些站是双模站时，这是必需的。在这种情况下，源发站地址可用于将命令确认定向回特定控制站，而不是整个系统。
     * 源发站地址将镜像 ASDU 和在监控方向（例如，通过一般询问进行询问）中询问的 ASDU 定向到激活该过程的源。
     * 如果未使用源发站地址（位设置为零）并且系统中定义了多个源，则必须将监控方向的 ASDU 定向到系统的所有相关源。在这种情况下，特定受影响的源必须选择其特定的 ASDU。
     *
     * @return 源发站地址 (ORG)
     */
    public byte getOrg() {
        return data[9];
    }

    public void setOrg(byte org) {
        data[9] = org;
    }

    /**
     * 该地址之所以称为公共地址，是因为它与 ASDU 中包含的所有对象相关联。这通常被解释为站地址，但是它可以构造为站/扇区地址，其中各个站被分解为多个逻辑单元。
     * COA 的长度为一个或两个八位字节，根据每个系统而固定。
     * 全局地址是指向特定系统的所有站的广播地址（广播地址）。在控制方向上具有广播地址的 ASDU 必须在监控方向上由特定定义的公共地址（站地址）的地址应答。根据标准，此参数由 2 个八位字节组成。
     * 未使用值 0，范围 1 – 65 534 表示站地址，值 65 535（0xFFFF）表示全局地址。
     * 当必须同时启动相同的应用程序功能时，使用全局地址。它仅限于以下 ASDU：
     * 类型=100（询问命令）：在公共时间回复特定系统数据快照
     * 类型=101（计数器询问命令）：在公共时间冻结总数
     * 类型=103（时钟同步命令）：将时钟同步到公共时间
     * 类型=105（重置过程命令）：同时重置
     *
     * @return ASDU 地址字段（ASDU 的通用地址，COA）
     */
    public int getCoa() {
        return ByteUtil.getShort(data, 10);
    }
    // endregion


}

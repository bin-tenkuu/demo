package demo.IEC104;

import demo.IEC104.content.BaseContent;
import lombok.val;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/10
 */
public class FrameUtil {

    /**
     * 根据帧类型预分配空间
     */
    public static Frame parse(FrameType type) {
        return switch (type) {
            case I -> new FrameI();
            case S -> new FrameS();
            case U -> new FrameU();
            default -> new Frame(type, new byte[6]);
        };
    }

    /**
     * 根据数据预分配帧空间，并将已有数据拷贝到帧中
     *
     * @param data length==2时返回{@link Frame}，length>2时返回具体类型
     */
    public static Frame parse(byte[] data) {
        val length = data.length;
        if (length < 2) {
            throw new IllegalArgumentException("帧长度不足");
        }
        if (data[0] != 0x68) {
            throw new IllegalArgumentException("帧起始符不正确");
        }
        val frameLength = (data[1] & 0xFF) + 2;
        val frameType = getFrameType(data);
        return switch (frameType) {
            case I -> {
                val bytes = new byte[12];
                val content = new byte[frameLength - 12];
                if (length < 12) {
                    System.arraycopy(data, 0, bytes, 0, length);
                } else {
                    System.arraycopy(data, 0, bytes, 0, 12);
                    System.arraycopy(data, 12, content, 0, length - 12);
                }
                yield new FrameI(bytes, content);
            }
            case S -> {
                val bytes = new byte[6];
                System.arraycopy(data, 0, bytes, 0, Math.min(6, length));
                yield new FrameS(bytes);
            }
            case U -> {
                val bytes = new byte[6];
                System.arraycopy(data, 0, bytes, 0, Math.min(6, length));
                yield new FrameU(bytes);
            }
            default -> {
                val bytes = new byte[frameLength];
                System.arraycopy(data, 0, bytes, 0, Math.min(frameLength, length));
                yield new Frame(frameType, bytes);
            }
        };

    }

    private static FrameType getFrameType(byte[] data) {
        if (data.length < 3) {
            return FrameType.UNKNOWN;
        }
        if ((data[2] & 1) == 0) {
            return FrameType.I;
        }
        if (data[2] == 1) {
            return FrameType.S;
        }
        if ((data[2] & 3) == 3) {
            return FrameType.U;
        }
        return FrameType.UNKNOWN;
    }

    public static byte[] initContent(FrameI frameI) {
        val typeID = TypeID.getByType(frameI.getTypeId());
        val number = frameI.getNumber();
        int size = 0;
        for (ContentLayout layout : typeID.layout) {
            size += layout.length;
        }
        if (frameI.getSq()) {
            size *= number;
            size += ContentLayout.IOA.length + typeID.timeLayout.length;
        } else {
            size += ContentLayout.IOA.length + typeID.timeLayout.length;
            size *= number;
        }
        val bytes = new byte[size];
        frameI.setContent(bytes);
        return bytes;
    }

    /**
     * 解析内容
     *
     * @return 对于sq == 1，会将ioa单独放入第一个list，时间戳放入最后一个list（如果有）
     */
    public static List<List<BaseContent>> parseContentsList(FrameI frameI) {
        val typeID = TypeID.getByType(frameI.getTypeId());
        val content = frameI.getContent();
        val number = frameI.getNumber();
        List<List<BaseContent>> contentList = new ArrayList<>(number);
        int offset = 0;
        List<BaseContent> contents;
        val contentLayouts = typeID.layout;
        if (frameI.getSq()) {
            contents = new ArrayList<>(1);
            offset = parseContentsList(ContentLayout.IOA, content, offset, contents);
            contentList.add(contents);
            for (int i = 0; i < number; i++) {
                contents = new ArrayList<>(contentLayouts.length);
                for (ContentLayout layout : contentLayouts) {
                    offset = parseContentsList(layout, content, offset, contents);
                }
                contentList.add(contents);
            }
            if (typeID.timeLayout != ContentLayout.NULL) {
                contents = new ArrayList<>(1);
                offset = parseContentsList(typeID.timeLayout, content, offset, contents);
                contentList.add(contents);
            }
        } else {
            while (offset < content.length) {
                contents = new ArrayList<>(contentLayouts.length + 1);
                offset = parseContentsList(ContentLayout.IOA, content, offset, contents);
                for (ContentLayout layout : contentLayouts) {
                    offset = parseContentsList(layout, content, offset, contents);
                }
                parseContentsList(typeID.timeLayout, content, offset, contents);
                contentList.add(contents);
            }
        }
        if (offset != content.length) {
            throw new IllegalArgumentException("内容长度不正确");
        }
        return contentList;
    }

    private static int parseContentsList(ContentLayout layout, byte[] content, int offset, List<BaseContent> list) {
        if (layout == ContentLayout.NULL) {
            return offset;
        }
        list.add(layout.parseContent(content, offset));
        return offset + layout.length;
    }

    public static byte[] toByteArray(List<List<BaseContent>> contentsList, boolean sq, ContentLayout timeLayout) {
        if (contentsList.isEmpty()) {
            return new byte[0];
        }
        val hasTime = timeLayout != ContentLayout.NULL;
        int size = contentsList.size();
        List<BaseContent> firstContents;
        if (sq) {
            size--;
            if (hasTime) {
                size--;
            }
            if (size == 0) {
                firstContents = null;
            } else {
                firstContents = contentsList.get(1);
            }
        } else {
            firstContents = contentsList.getFirst();
        }
        int index = 0;
        if (firstContents != null) {
            for (BaseContent content : firstContents) {
                index += content.size();
            }
        }
        int totalSize = index * size;
        if (sq) {
            totalSize += ContentLayout.IOA.length + timeLayout.length;
        }
        val bytes = new byte[totalSize];
        index = 0;
        for (List<BaseContent> contents : contentsList) {
            for (BaseContent content : contents) {
                content.writeTo(bytes, index);
                index += content.size();
            }
        }
        return bytes;
    }

    // region toString
    public static String toString(Frame frame) {
        val sb = new StringBuilder();
        toString(sb, frame);
        return sb.toString();
    }

    public static void toString(StringBuilder sb, Frame frame) {
        val data = frame.data;
        sb.append("类型：").append(frame.type.name()).append(" 帧\n");
        sb.append(ByteUtil.toString(data[0])).append(" <起始符> ");
        sb.append(ByteUtil.toString(data[1])).append(" <长度>\n");
        ByteUtil.toString(sb, data, 2, 2);
        switch (frame) {
            case FrameU _ -> {
                sb.append("<控制功能> ");
                ByteUtil.toString(sb, data, 4, 2);
            }
            case FrameS _ -> {
                ByteUtil.toString(sb, data, 4, 2);
                sb.append("<接收序号>");
            }
            case FrameI i -> {
                sb.append("<发送序号> ");
                ByteUtil.toString(sb, data, 4, 2);
                sb.append("<接收序号>\n");
                val typeID = TypeID.getByType(i.getTypeId());
                sb.append(ByteUtil.toString(data[6])).append(" <类型标志：").append(typeID.name).append("> ");
                sb.append(ByteUtil.toString(data[7])).append(" <地址");
                if (i.getSq()) {
                    sb.append("连续,");
                } else {
                    sb.append("不连续,");
                }
                sb.append(i.getNumber()).append("个对象>\n");
                sb.append(ByteUtil.toString(data[8])).append(" <");
                if (i.getT()) {
                    sb.append("测试,");
                } else {
                    sb.append("非测试，");
                }
                if (i.getPn()) {
                    sb.append("消极，");
                } else {
                    sb.append("积极，");
                }
                val cot = CauseOfTransmission.getByType(i.getCot());
                sb.append("传送原因：").append(cot.name).append("> ");
                sb.append(ByteUtil.toString(data[9])).append(" <源发站地址>\n");
                ByteUtil.toString(sb, data, 10, 2);
                sb.append("<通用地址>\n");
                val contentsList = i.getContentsList();
                for (val contents : contentsList) {
                    for (val content : contents) {
                        if (content != null) {
                            ByteUtil.toString(sb, content.toByteArray());
                            sb.append("<").append(content.getClass().getSimpleName()).append(":")
                                    .append(content).append("> ");
                        }
                    }
                    sb.append("\n");
                }

            }
            default -> ByteUtil.toString(sb, data, 4, data.length - 4);
        }
        sb.append("\n");
    }
    // endregion

    public static void main(String[] args) {
        // https://blog.redisant.cn/docs/iec104-tutorial/chapter8/
        System.out.println("\n总召唤流程详解");
        printfromString("68-04-07-00-00-00");
        printfromString("68-04-0B-00-00-00");
        printfromString("68-0E-00-00-00-00-64-01-06-00-01-00-00-00-00-14");
        printfromString("68-0E-00-00-02-00-64-01-07-00-01-00-00-00-00-14");
        printfromString("68-12-02-00-02-00-01-02-14-00-01-00-01-00-00-00-02-00-00-00");
        printfromString("68-12-04-00-02-00-03-02-14-00-01-00-01-00-00-00-02-00-00-00");
        printfromString("68-0E-06-00-02-00-64-01-0A-00-01-00-00-00-00-14");
        // https://blog.redisant.cn/docs/iec104-tutorial/chapter9/
        System.out.println("\n计数量召唤流程详解");
        printfromString("68-04-07-00-00-00");
        printfromString("68-04-0B-00-00-00");
        printfromString("68-0E-00-00-00-00-65-01-06-00-01-00-00-00-00-05");
        printfromString("68-0E-00-00-02-00-65-01-07-00-01-00-00-00-00-05");
        printfromString("68-1A-02-00-02-00-0F-02-25-00-01-00-01-00-00-00-00-00-00-00-02-00-00-00-00-00-00-00");
        printfromString("68-0E-04-00-02-00-65-01-0A-00-01-00-00-00-00-05");
        System.out.println("\n其他");
        printfromString("""
                68 1e 04 00 02 00 03 05 14 00 01 00 01 00 00 02 06 00 00 02 0a 00 00 01 0b 00 00 02 0c 00 00 01""");
        printfromString("68 13 06 00 02 00 09 82 14 00 01 00 01 07 00 a1 10 00 89 15 00");
        printfromString("68 14 02 00 0a 00 67 01 06 00 01 00 00 00 00 16 23 32 10 13 05 08");
        printfromString("""
                68 3A 76 67 78 16 0F 06 03 00 01 00 06 64 00 45 47 09 00 00 4B 64 00 CF A2 00 00 00 4E 64 00 CF A2 00 00
                00 5B 64 00 41 7A 00 00 00 5F 64 00 41 7A 00 00 00 72 64 00 14 6A 00 00 00""");
        printfromString("""
                68 2A 04 00 02 00 0D 04 14 00 01 00 01 40 00 00 78 DB 3F 00 02 40 00 00 D8 90 42 00 03 40 00 00 F4 92 42
                00 04 40 00 60 50 9A 3F 00""");
        printfromString("68 12 0E 00 10 00 0D 01 03 00 01 00 02 40 00 00 78 DB 3F 00");
        printfromString("68 1A 02 00 02 00 03 04 14 00 01 00 01 00 00 01 02 00 00 02 03 00 00 01 04 00 00 02");
        printfromString("""
                68 d5 cc 07 fc 20 0f a8 25 00 01 00 01 64 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
                a1 0d 00 00 00 be 01 00 00 00 5f 0f 00 00 00 8e 05 00 00 00 c0 03 00 00 00 05 00 00 00 00 c5 03 00 00 00
                15 03 00 00 00 0b 00 00 00 00 94 01 00 00 00 9f 01 00 00 00 88 01 00 00 00 05 00 00 00 00 67 00 00 00 00
                6d 00 00 00 00 4b 00 00 00 00 d9 02 00 00 00 1b 00 00 00 00 f4 02 00 00 00 dd 01 00 00 00 00 00 00 00 00
                00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 29 0a 00 00 00 85 01 00 00 00 ae 0b 00 00 00 79 09 00 00 00
                3b 06 00 00 00 11 00 00 00 00 4d 06 00 00 00 cf 03 00 00 00 d7 10 00 00 00 8e 01 00 00 00 66 12 00 00 00
                c0 08 00 00 00""");
        printfromString("68-0E-20-00-5C-00-2D-01-08-00-01-00-03-60-00-01");
        printfromString("68-0E-5C-00-22-00-2D-01-6D-00-01-00-03-60-00-01");
    }

    private static void printfromString(String string) {
        // print(parse(ByteUtil.fromString(string)));
        System.out.println(toString(parse(ByteUtil.fromString(string))));
    }

    private static void print(Frame frame) {
        val data = frame.getData();
        System.out.println(ByteUtil.toString(data));
        val sb = new StringBuilder();
        sb.append(frame.type).append(": ");
        sb.append("length=").append(frame.getLength() + 2).append("/").append(data.length).append(", ");
        if (frame.getLength() + 2 != data.length) {
            System.out.println(sb);
            sb.setLength(0);
            System.err.println("解析Length不正确");
        }
        switch (frame) {
            case FrameU frameU -> {
                sb.append("test=").append(frameU.getTest()).append(", ");
                sb.append("start=").append(frameU.getStart()).append(", ");
                sb.append("stop=").append(frameU.getStop()).append(", ");
            }
            case FrameS frameS -> sb.append("receiveCounte=").append(frameS.getReceiveCounte()).append(", ");
            case FrameI frameI -> {
                sb.append("sendCounte=").append(frameI.getSendCounte()).append(", ");
                sb.append("receiveCounte=").append(frameI.getReceiveCounte()).append(", ");
                val typeID = TypeID.getByType(frameI.getTypeId());
                sb.append("TypeId=").append(typeID.name).append("(").append(typeID.type).append(')').append(", ");
                sb.append("SQ=").append(frameI.getSq()).append(", ");
                val number = frameI.getNumber();
                sb.append("number=").append(number).append(", ");
                sb.append("T=").append(frameI.getT()).append(", ");
                sb.append("P/N=").append(frameI.getPn()).append(", ");
                val cot = CauseOfTransmission.getByType(frameI.getCot());
                sb.append("COT=").append(cot.name).append("(").append(cot.type).append(')').append(", ");
                sb.append("ORG=").append(frameI.getOrg()).append(", ");
                sb.append("COA=").append(frameI.getCoa()).append(", ");
                int contentLength = frameI.getSq() ? 0 : ContentLayout.IOA.length + typeID.timeLayout.length;
                for (ContentLayout layout : typeID.layout) {
                    contentLength += layout.length;
                }
                sb.append("contentLength=").append(contentLength).append(", ");
                val contentsList = frameI.getContentsList();
                for (int i = 0; i < contentsList.size(); i++) {
                    val contents = contentsList.get(i);
                    sb.append("\ncontent").append(i).append(":\t");
                    for (val content : contents) {
                        if (content != null) {
                            sb.append(content).append(", ");
                        }
                    }
                }
                val bytes = toByteArray(contentsList, frameI.getSq(), typeID.timeLayout);
                val equals = Arrays.equals(frameI.getContent(), bytes);
                if (!equals) {
                    System.out.println(sb);
                    sb.setLength(0);
                    System.err.println("Content内容不一致");
                }
            }
            default -> {
            }
        }
        val bytes = frame.toByteArray();
        val equals = Arrays.equals(data, bytes);
        if (!equals) {
            System.err.println("Data内容不一致");
        }
        System.out.println(sb);
        System.out.println();
    }
}

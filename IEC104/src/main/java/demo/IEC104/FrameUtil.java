package demo.IEC104;

import demo.IEC104.content.BaseContent;
import lombok.val;
import org.jetbrains.annotations.NotNull;

import java.nio.ByteBuffer;
import java.util.ArrayList;
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
    public static Frame build(FrameType type) {
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
     * @param buffer 长度足够时返回{@link Frame}，不足时返回null
     */
    public static Frame parse(ByteBuffer buffer) {
        buffer.flip();
        var length = buffer.remaining();
        if (length < 6) {
            System.out.printf("帧长度不足: %d\n", length);
            buffer.compact();
            return null;
        }
        if (buffer.get(0) != 0x68) {
            System.out.printf("帧起始符不正确: %s\n", ByteUtil.toString(buffer.get(0)));
            buffer.compact();
            return null;
        }
        var frameLength = (buffer.get(1) & 0xFF) + 2;
        if (length < frameLength) {
            System.out.printf("帧长度不足: %d < %d\n", length, frameLength);
            buffer.compact();
            return null;
        }
        var frameType = getFrameType(buffer.get(2));
        var position = buffer.position();
        var frame = switch (frameType) {
            case I -> {
                var bytes = new byte[12];
                var content = new byte[frameLength - 12];
                buffer.get(position, bytes, 0, 12);
                buffer.get(position + 12, content, 0, content.length);
                yield new FrameI(bytes, content);
            }
            case S -> {
                var bytes = new byte[6];
                buffer.get(position, bytes, 0, 6);
                yield new FrameS(bytes);
            }
            case U -> {
                var bytes = new byte[6];
                buffer.get(position, bytes, 0, 6);
                yield new FrameU(bytes);
            }
            default -> {
                var bytes = new byte[frameLength];
                buffer.get(position, bytes, 0, frameLength);
                yield new Frame(frameType, bytes);
            }
        };
        buffer.position(position + frameLength);
        buffer.compact();
        return frame;
    }

    /**
     * 根据数据预分配帧空间，并将已有数据拷贝到帧中
     *
     * @param data length==2时返回{@link Frame}，length>2时返回具体类型
     */
    public static Frame parse(byte[] data) {
        var length = data.length;
        if (length < 6) {
            throw new IllegalArgumentException("帧长度不足: " + length);
        }
        if (data[0] != 0x68) {
            throw new IllegalArgumentException("帧起始符不正确: " + ByteUtil.toString(data[0]));
        }
        var frameLength = (data[1] & 0xFF) + 2;
        if (length < frameLength) {
            throw new IllegalArgumentException("帧长度不足: " + length + " < " + frameLength);
        }
        var frameType = getFrameType(data[2]);
        return switch (frameType) {
            case I -> {
                var bytes = new byte[12];
                var content = new byte[frameLength - 12];
                System.arraycopy(data, 0, bytes, 0, 12);
                System.arraycopy(data, 12, content, 0, content.length);
                yield new FrameI(bytes, content);
            }
            case S -> {
                var bytes = new byte[6];
                System.arraycopy(data, 0, bytes, 0, 6);
                yield new FrameS(bytes);
            }
            case U -> {
                var bytes = new byte[6];
                System.arraycopy(data, 0, bytes, 0, 6);
                yield new FrameU(bytes);
            }
            default -> {
                val bytes = new byte[frameLength];
                System.arraycopy(data, 0, bytes, 0, frameLength);
                yield new Frame(frameType, bytes);
            }
        };
    }

    private static FrameType getFrameType(byte data) {
        if ((data & 1) == 0) {
            return FrameType.I;
        }
        if (data == 1) {
            return FrameType.S;
        }
        if ((data & 3) == 3) {
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
    @NotNull
    public static List<@NotNull List<BaseContent>> parseContentsList(FrameI frameI) {
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
        if (frame.getLength() + 2 != frame.getDataLength()) {
            System.err.printf("报文长度不正确：%s/%s\n", frame.getLength() + 2, frame.getDataLength());
        }
        ByteUtil.toString(sb, data, 2, 2);
        switch (frame) {
            case FrameU u -> {
                sb.append("<控制功能：");
                int control;
                if ((control = u.getTest()) != 0) {
                    sb.append("测试");
                } else if ((control = u.getStart()) != 0) {
                    sb.append("启动");
                } else if ((control = u.getStop()) != 0) {
                    sb.append("停止");
                }
                if (control == 2) {
                    sb.append("确认");
                }
                sb.append("> ");
                ByteUtil.toString(sb, data, 4, 2);
            }
            case FrameS s -> {
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
                if (!i.getSq()) {
                    sb.append("不");
                }
                sb.append("连续，");
                sb.append(i.getNumber()).append("个对象>\n");
                sb.append(ByteUtil.toString(data[8])).append(" <");
                if (!i.getT()) {
                    sb.append("非");
                }
                sb.append("测试，");
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
                        ByteUtil.toString(sb, content.toByteArray());
                        sb.append("<").append(content.getClass().getSimpleName()).append(":");
                        content.toString(sb);
                        sb.append("> ");
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
        printFromString("68-04-07-00-00-00");
        printFromString("68-04-0B-00-00-00");
        printFromString("68-0E-00-00-00-00-64-01-06-00-01-00-00-00-00-14");
        printFromString("68-0E-00-00-02-00-64-01-07-00-01-00-00-00-00-14");
        printFromString("68-12-02-00-02-00-01-02-14-00-01-00-01-00-00-00-02-00-00-00");
        printFromString("68-12-04-00-02-00-03-02-14-00-01-00-01-00-00-00-02-00-00-00");
        printFromString("68-0E-06-00-02-00-64-01-0A-00-01-00-00-00-00-14");
        // https://blog.redisant.cn/docs/iec104-tutorial/chapter9/
        System.out.println("\n计数量召唤流程详解");
        printFromString("68-04-07-00-00-00");
        printFromString("68-04-0B-00-00-00");
        printFromString("68-0E-00-00-00-00-65-01-06-00-01-00-00-00-00-05");
        printFromString("68-0E-00-00-02-00-65-01-07-00-01-00-00-00-00-05");
        printFromString("68-1A-02-00-02-00-0F-02-25-00-01-00-01-00-00-00-00-00-00-00-02-00-00-00-00-00-00-00");
        printFromString("68-0E-04-00-02-00-65-01-0A-00-01-00-00-00-00-05");
        System.out.println("\n其他");
        printFromString("""
                68 1e 04 00 02 00 03 05 14 00 01 00 01 00 00 02 06 00 00 02 0a 00 00 01 0b 00 00 02 0c 00 00 01""");
        printFromString("68 13 06 00 02 00 09 82 14 00 01 00 01 07 00 a1 10 00 89 15 00");
        printFromString("""
                68 3A 76 67 78 16 0F 06 03 00 01 00 06 64 00 45 47 09 00 00 4B 64 00 CF A2 00 00 00 4E 64 00 CF A2 00 00
                00 5B 64 00 41 7A 00 00 00 5F 64 00 41 7A 00 00 00 72 64 00 14 6A 00 00 00""");
        printFromString("""
                68 2A 04 00 02 00 0D 04 14 00 01 00 01 40 00 00 78 DB 3F 00 02 40 00 00 D8 90 42 00 03 40 00 00 F4 92 42
                00 04 40 00 60 50 9A 3F 00""");
        printFromString("68 12 0E 00 10 00 0D 01 03 00 01 00 02 40 00 00 78 DB 3F 00");
        printFromString("68 1A 02 00 02 00 03 04 14 00 01 00 01 00 00 01 02 00 00 02 03 00 00 01 04 00 00 02");
        printFromString("""
                68 d5 cc 07 fc 20 0f a8 25 00 01 00 01 64 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
                a1 0d 00 00 00 be 01 00 00 00 5f 0f 00 00 00 8e 05 00 00 00 c0 03 00 00 00 05 00 00 00 00 c5 03 00 00 00
                15 03 00 00 00 0b 00 00 00 00 94 01 00 00 00 9f 01 00 00 00 88 01 00 00 00 05 00 00 00 00 67 00 00 00 00
                6d 00 00 00 00 4b 00 00 00 00 d9 02 00 00 00 1b 00 00 00 00 f4 02 00 00 00 dd 01 00 00 00 00 00 00 00 00
                00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 29 0a 00 00 00 85 01 00 00 00 ae 0b 00 00 00 79 09 00 00 00
                3b 06 00 00 00 11 00 00 00 00 4d 06 00 00 00 cf 03 00 00 00 d7 10 00 00 00 8e 01 00 00 00 66 12 00 00 00
                c0 08 00 00 00""");
        printFromString("68-0E-20-00-5C-00-2D-01-08-00-01-00-03-60-00-01");
        printFromString("68-0E-5C-00-22-00-2D-01-6D-00-01-00-03-60-00-01");
    }

    private static void printFromString(String string) {
        System.out.println(toString(parse(ByteUtil.fromString(string))));
    }

}

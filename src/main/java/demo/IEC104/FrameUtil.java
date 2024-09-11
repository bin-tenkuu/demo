package demo.IEC104;

import lombok.val;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/10
 */
public class FrameUtil {

    public static Frame parse(FrameType type) {
        return switch (type) {
            case I -> new FrameI();
            case S -> new FrameS();
            case U -> new FrameU();
            default -> new Frame(type, new byte[6]);
        };
    }

    public static Frame parse(byte[] data) {
        if (data.length < 6) {
            throw new IllegalArgumentException("帧长度不足");
        }
        if (data[0] != 0x68) {
            throw new IllegalArgumentException("帧起始符不正确");
        }
        val frameType = getFrameType(data);
        return switch (frameType) {
            case I -> new FrameI(frameType, data);
            case S -> new FrameS(frameType, data);
            case U -> new FrameU(frameType, data);
            default -> new Frame(frameType, data);
        };
    }

    private static FrameType getFrameType(byte[] data) {
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

    private static List<Map<String, Object>> parseContent(FrameI frameI) {
        val typeID = TypeID.getByType(frameI.getTypeId());
        val content = frameI.getContent();
        val number = frameI.getNumber();
        val list = new ArrayList<Map<String, Object>>(number);
        int offset = 0;
        Map<String, Object> map;
        if (frameI.getSq()) {
            val first = map = new LinkedHashMap<>();
            offset = parseContent(ContentLayout.IOA, content, offset, map);
            for (int i = 0; i < number; i++) {
                for (ContentLayout layout : typeID.layout) {
                    offset = parseContent(layout, content, offset, map);
                }
                list.add(map);
                map = new LinkedHashMap<>();
            }
            offset = parseContent(typeID.timeLayout, content, offset, first);
        } else {
            while (offset < content.length) {
                map = new LinkedHashMap<>();
                offset = parseContent(ContentLayout.IOA, content, offset, map);
                for (ContentLayout layout : typeID.layout) {
                    offset = parseContent(layout, content, offset, map);
                }
                parseContent(typeID.timeLayout, content, offset, map);
                list.add(map);
            }
        }
        if (offset != content.length) {
            throw new IllegalArgumentException("内容长度不正确");
        }
        return list;
    }

    private static int parseContent(ContentLayout layout, byte[] content, int offset, Map<String, Object> map) {
        layout.parseContent(content, offset, map);
        return offset + layout.length;
    }

    private static Map<String, Object> parseContent(byte[] content, TypeID typeID) {
        int of = 0;
        val map = new LinkedHashMap<String, Object>();
        ContentLayout.IOA.parseContent(content, of, map);
        for (ContentLayout layout : typeID.layout) {
            layout.parseContent(content, of, map);
            of += layout.length;
        }
        typeID.timeLayout.parseContent(content, of, map);
        return map;
    }

    public static void main(String[] args) {
        System.out.println((byte) 0b11000000 << 1 >> 1);
        // https://blog.redisant.cn/docs/iec104-tutorial/chapter8/
        System.out.println("\n\n总召唤流程详解");
        printfromString("68-04-07-00-00-00");
        printfromString("68-04-0B-00-00-00");
        printfromString("68-0E-00-00-00-00-64-01-06-00-01-00-00-00-00-14");
        printfromString("68-0E-00-00-02-00-64-01-07-00-01-00-00-00-00-14");
        printfromString("68-12-02-00-02-00-01-02-14-00-01-00-01-00-00-00-02-00-00-00");
        printfromString("68-12-04-00-02-00-03-02-14-00-01-00-01-00-00-00-02-00-00-00");
        printfromString("68-0E-06-00-02-00-64-01-0A-00-01-00-00-00-00-14");
        // https://blog.redisant.cn/docs/iec104-tutorial/chapter9/
        System.out.println("\n\n计数量召唤流程详解");
        printfromString("68-04-07-00-00-00");
        printfromString("68-04-0B-00-00-00");
        printfromString("68-0E-00-00-00-00-65-01-06-00-01-00-00-00-00-05");
        printfromString("68-0E-00-00-02-00-65-01-07-00-01-00-00-00-00-05");
        printfromString("68-1A-02-00-02-00-0F-02-25-00-01-00-01-00-00-00-00-00-00-00-02-00-00-00-00-00-00-00");
        printfromString("68-0E-04-00-02-00-65-01-0A-00-01-00-00-00-00-05");
        System.out.println("\n\n其他");
        printfromString(
                "68 1e 04 00 02 00 03 05 14 00 01 00 01 00 00 02 06 00 00 02 0a 00 00 01 0b 00 00 02 0c 00 00 01");
        printfromString(
                "68 13 06 00 02 00 09 82 14 00 01 00 01 07 00 a1 10 00 89 15 00");
        printfromString(
                "68 14 02 00 0a 00 67 01 06 00 01 00 00 00 00 01 02 03 04 81 09 05");
        printfromString(
                "68 3A 76 67 78 16 0F 06 03 00 01 00 06 64 00 45 47 09 00 00 4B 64 00 CF A2 00 00 00 4E 64 00 CF A2 00 00 00 5B 64 00 41 7A 00 00 00 5F 64 00 41 7A 00 00 00 72 64 00 14 6A 00 00 00");
        printfromString(
                "68 2A 04 00 02 00 0D 04 14 00 01 00 01 40 00 00 78 DB 3F 00 02 40 00 00 D8 90 42 00 03 40 00 00 F4 92 42 00 04 40 00 60 50 9A 3F 00");
        printfromString("68 12 0E 00 10 00 0D 01 03 00 01 00 02 40 00 00 78 DB 3F 00");
    }

    private static void printfromString(String string) {
        print(parse(ByteUtil.fromString(string)));
    }

    private static void print(Frame frame) {
        System.out.println();
        val data = frame.getData();
        System.out.println(ByteUtil.toString(data));
        val sb = new StringBuilder();
        sb.append(frame.type).append(": ");
        sb.append("length=").append(frame.getLength() + 2).append("/").append(data.length).append(", ");
        switch (frame) {
            case FrameU frameU -> {
                sb.append("test=").append(frameU.getTest()).append(", ");
                sb.append("start=").append(frameU.getStart()).append(", ");
                sb.append("stop=").append(frameU.getStop()).append(", ");
            }
            case FrameS frameS -> sb.append("receiveCounter=").append(frameS.getReceiveCounter()).append(", ");
            case FrameI frameI -> {
                sb.append("sendCounter=").append(frameI.getSendCounter()).append(", ");
                sb.append("receiveCounter=").append(frameI.getReceiveCounter()).append(", ");
                val typeID = TypeID.getByType(frameI.getTypeId());
                sb.append("TypeId=").append(typeID.name).append(", ");
                sb.append("SQ=").append(frameI.getSq()).append(", ");
                val number = frameI.getNumber();
                sb.append("number=").append(number).append(", ");
                sb.append("T=").append(frameI.getT()).append(", ");
                sb.append("P/N=").append(frameI.getPn()).append(", ");
                sb.append("COT=").append(CauseOfTransmission.getByType(frameI.getCot()).name).append(", ");
                sb.append("ORG=").append(frameI.getOrg()).append(", ");
                sb.append("COA=").append(frameI.getCoa()).append(", ");
                val content = frameI.getContent();
                int contentLength = 3 + typeID.timeLayout.length;
                for (ContentLayout layout : typeID.layout) {
                    contentLength += layout.length;
                }
                sb.append("contentLength=").append(contentLength).append(", ");
                val maps = parseContent(frameI);
                for (int i = 0; i < maps.size(); i++) {
                    val map = maps.get(i);
                    sb.append("\ncontent").append(i).append(":\t");
                    for (val entry : map.entrySet()) {
                        sb.append(entry.getKey()).append("=").append(entry.getValue()).append(", ");
                    }
                }
            }
            default -> {
            }
        }
        System.out.println(sb);
    }
}

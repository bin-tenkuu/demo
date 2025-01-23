package demo.IEC104.sc;

import demo.IEC104.Frame;
import demo.IEC104.FrameI;
import demo.IEC104.FrameS;
import demo.IEC104.FrameU;
import lombok.RequiredArgsConstructor;

import java.util.function.Consumer;

/**
 * @author bin
 * @version 1.0.0
 * @since 2025/01/23
 */
@RequiredArgsConstructor
public class FrameResponse implements Consumer<Frame> {
    private final Client client;

    private int sendCounter = 0;
    private int receiverCounte = 0;

    public void accept(Frame frame) {
        switch (frame) {
            case FrameI I -> {
                receiverCounte = I.getSendCounte();
            }
            case FrameS S -> {
                receiverCounte = S.getReceiveCounte();
            }
            case FrameU U -> {
                if (U.getTest() == 1) {
                    U.setTest(2);
                    client.write(U);
                } else if (U.getStart() == 1) {
                    U.setStart(2);
                    client.write(U);
                } else if (U.getStop() == 1) {
                    U.setStop(2);
                    client.write(U);
                }
            }
            default -> {

            }
        }
    }

}

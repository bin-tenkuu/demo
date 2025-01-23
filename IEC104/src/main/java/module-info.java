/**
 * @author bin
 * @since 2024/07/23
 */
module demo {
    requires static lombok;
    requires static org.jetbrains.annotations;

    exports demo.IEC104;
    exports demo.IEC104.content;
    exports demo.IEC104.sc;
}

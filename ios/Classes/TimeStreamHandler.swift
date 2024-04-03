//
//  TimeStreamHandler.swift
//  ultralytics_yolo
//
//  Created by Sergio Sánchez on 9/11/23.
//

import Foundation

class TimeStreamHandler: NSObject, FlutterStreamHandler {
    private let handler = DispatchQueue.main
    private var eventSink: FlutterEventSink?

    func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        eventSink = events
        return nil
    }
    
    func onCancel(withArguments arguments: Any?) -> FlutterError? {
        eventSink = nil
        return nil
    }
    
    func sink(time: Double) {
        handler.async {
            if let eventSink = self.eventSink {
                eventSink(time)
            }
        }
    }
    
    func close() {
        if let eventSink = eventSink {
            eventSink(FlutterEndOfEventStream)
            self.eventSink = nil
        }
    }
}
